use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::sync::mpsc::{self, Receiver, TrySendError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_QUEUED_MESSAGES: usize = 64;

/// Bounded background reader for Content-Length framed JSON-RPC streams. Pipe reads happen only
/// on the reader thread, so callers can enforce an actual deadline even when a server is silent.
pub(crate) struct FramedJsonReader {
    protocol: &'static str,
    receiver: Receiver<Result<Value, String>>,
    thread: Option<JoinHandle<()>>,
}

/// Bounded background reader for newline-delimited JSON-RPC streams such as MCP stdio. MCP
/// messages must occupy exactly one line; this deliberately does not accept LSP framing.
pub(crate) struct JsonLineReader {
    protocol: &'static str,
    receiver: Receiver<Result<Value, String>>,
    thread: Option<JoinHandle<()>>,
}

impl JsonLineReader {
    pub(crate) fn spawn(
        protocol: &'static str,
        reader: impl Read + Send + 'static,
        max_message_bytes: usize,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(MAX_QUEUED_MESSAGES);
        let thread = std::thread::Builder::new()
            .name(format!("pb-{}-stdout", protocol.to_ascii_lowercase()))
            .spawn(move || {
                let mut reader = BufReader::new(reader);
                loop {
                    let message = read_json_line(&mut reader, protocol, max_message_bytes)
                        .map_err(|error| format!("{error:#}"));
                    let failed = message.is_err();
                    match sender.try_send(message) {
                        Ok(()) if !failed => {}
                        Ok(()) => break,
                        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => break,
                    }
                }
            })
            .with_context(|| format!("failed to start {protocol} stdout reader"))?;
        Ok(Self {
            protocol,
            receiver,
            thread: Some(thread),
        })
    }

    pub(crate) fn recv_until(&self, deadline: Instant) -> Result<Value> {
        recv_until(&self.receiver, self.protocol, deadline)
    }

    pub(crate) fn join(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl FramedJsonReader {
    pub(crate) fn spawn(
        protocol: &'static str,
        reader: impl Read + Send + 'static,
        max_body_bytes: usize,
    ) -> Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(MAX_QUEUED_MESSAGES);
        let thread = std::thread::Builder::new()
            .name(format!("pb-{}-stdout", protocol.to_ascii_lowercase()))
            .spawn(move || {
                let mut reader = BufReader::new(reader);
                loop {
                    let message =
                        read_content_length_message(&mut reader, protocol, max_body_bytes)
                            .map_err(|error| format!("{error:#}"));
                    let failed = message.is_err();
                    match sender.try_send(message) {
                        Ok(()) if !failed => {}
                        Ok(()) => break,
                        Err(TrySendError::Full(_)) => break,
                        Err(TrySendError::Disconnected(_)) => break,
                    }
                }
            })
            .with_context(|| format!("failed to start {protocol} stdout reader"))?;
        Ok(Self {
            protocol,
            receiver,
            thread: Some(thread),
        })
    }

    pub(crate) fn recv_until(&self, deadline: Instant) -> Result<Value> {
        recv_until(&self.receiver, self.protocol, deadline)
    }

    pub(crate) fn try_recv(&self) -> Result<Option<Value>> {
        match self.receiver.try_recv() {
            Ok(Ok(message)) => Ok(Some(message)),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                bail!("{} response stream closed", self.protocol)
            }
        }
    }

    /// Join only after the owning process has been stopped and its stdout pipe has closed.
    pub(crate) fn join(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn recv_until(
    receiver: &Receiver<Result<Value, String>>,
    protocol: &str,
    deadline: Instant,
) -> Result<Value> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or(Duration::ZERO);
    if remaining.is_zero() {
        bail!("timed out waiting for {protocol} response");
    }
    match receiver.recv_timeout(remaining) {
        Ok(Ok(message)) => Ok(message),
        Ok(Err(error)) => Err(anyhow!(error)),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            bail!("timed out waiting for {protocol} response")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            bail!("{protocol} response stream closed")
        }
    }
}

fn read_json_line(
    reader: &mut impl BufRead,
    protocol: &str,
    max_message_bytes: usize,
) -> Result<Value> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            bail!("{protocol} server exited before sending a complete response");
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if line.len().saturating_add(take) > max_message_bytes.saturating_add(1) {
            bail!("{protocol} response is too large: more than {max_message_bytes} bytes");
        }
        let complete = available.get(take.saturating_sub(1)) == Some(&b'\n');
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if complete {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.len() > max_message_bytes {
                bail!("{protocol} response is too large: {} bytes", line.len());
            }
            return serde_json::from_slice(&line)
                .with_context(|| format!("failed to parse {protocol} JSON-line response"));
        }
    }
}

fn read_content_length_message(
    reader: &mut impl BufRead,
    protocol: &str,
    max_body_bytes: usize,
) -> Result<Value> {
    let mut content_length = None;
    let mut header_bytes = 0usize;
    loop {
        let line = read_bounded_header_line(
            reader,
            MAX_HEADER_BYTES.saturating_sub(header_bytes),
            protocol,
        )?;
        let read = line.len();
        header_bytes = header_bytes.saturating_add(read);
        let line = String::from_utf8(line).context("JSON-RPC response header is not UTF-8")?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("Content-Length")
        {
            let length = value
                .trim()
                .parse::<usize>()
                .context("invalid JSON-RPC Content-Length header")?;
            if length > max_body_bytes {
                bail!("{protocol} response is too large: {length} bytes");
            }
            content_length = Some(length);
        }
    }
    let length = content_length.context("JSON-RPC response missing Content-Length header")?;
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body).context("failed to parse JSON-RPC response JSON")
}

fn read_bounded_header_line(
    reader: &mut impl BufRead,
    remaining: usize,
    protocol: &str,
) -> Result<Vec<u8>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            bail!("{protocol} server exited before sending a complete response");
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if line.len().saturating_add(take) > remaining {
            bail!("{protocol} response headers exceed {MAX_HEADER_BYTES} bytes");
        }
        let complete = available.get(take.saturating_sub(1)) == Some(&b'\n');
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if complete {
            return Ok(line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn framed_reader_parses_messages_and_reports_stream_close() {
        let body = br#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        let framed = format!("Content-Length: {}\r\n\r\n", body.len())
            .into_bytes()
            .into_iter()
            .chain(body.iter().copied())
            .collect::<Vec<_>>();
        let mut reader = FramedJsonReader::spawn("TEST", Cursor::new(framed), 1024).unwrap();
        assert_eq!(
            reader
                .recv_until(Instant::now() + Duration::from_secs(1))
                .unwrap()["id"],
            1
        );
        assert!(
            reader
                .recv_until(Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .to_string()
                .contains("exited")
        );
        reader.join();
    }

    #[test]
    fn framed_reader_enforces_real_deadline_and_body_bound() {
        let (_sender, receiver) = mpsc::sync_channel(1);
        let framed = FramedJsonReader {
            protocol: "TEST",
            receiver,
            thread: None,
        };
        let timeout = framed
            .recv_until(Instant::now() + Duration::from_millis(10))
            .unwrap_err()
            .to_string();
        assert!(timeout.contains("timed out"));

        let mut oversized = FramedJsonReader::spawn(
            "TEST",
            Cursor::new(b"Content-Length: 9\r\n\r\n123456789"),
            8,
        )
        .unwrap();
        assert!(
            oversized
                .recv_until(Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .to_string()
                .contains("too large")
        );
        oversized.join();
    }

    #[test]
    fn json_line_reader_uses_mcp_framing_and_enforces_the_bound() {
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n";
        let mut reader = JsonLineReader::spawn("MCP", Cursor::new(input), 1024).unwrap();
        assert_eq!(
            reader
                .recv_until(Instant::now() + Duration::from_secs(1))
                .unwrap()["id"],
            1
        );
        reader.join();

        let mut oversized =
            JsonLineReader::spawn("MCP", Cursor::new(b"{\"too\":\"large\"}\n"), 8).unwrap();
        assert!(
            oversized
                .recv_until(Instant::now() + Duration::from_secs(1))
                .unwrap_err()
                .to_string()
                .contains("too large")
        );
        oversized.join();
    }
}
