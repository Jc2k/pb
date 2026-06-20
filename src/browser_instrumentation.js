if (!window.__pbBrowserDebug) {
  const state = { console: [], network: [] };
  Object.defineProperty(window, '__pbBrowserDebug', { value: state, configurable: false });
  for (const level of ['log', 'info', 'warn', 'error']) {
    const original = console[level];
    console[level] = function(...args) {
      state.console.push({ level, message: args.map(String).join(' '), timestamp: Date.now() });
      return original.apply(this, args);
    };
  }
  window.addEventListener('error', e => state.console.push({ level: 'error', message: e.message, source: e.filename, line: e.lineno, column: e.colno, timestamp: Date.now() }));
  window.addEventListener('unhandledrejection', e => state.console.push({ level: 'error', message: 'Unhandled promise rejection: ' + String(e.reason), timestamp: Date.now() }));
  const originalFetch = window.fetch;
  if (originalFetch) {
    window.fetch = async function(input, init) {
      const started = performance.now();
      const url = typeof input === 'string' ? input : input && input.url;
      const record = { kind: 'fetch', url, method: (init && init.method) || 'GET', startedAt: Date.now() };
      state.network.push(record);
      try {
        const response = await originalFetch.apply(this, arguments);
        record.status = response.status;
        record.ok = response.ok;
        record.durationMs = performance.now() - started;
        return response;
      } catch (err) {
        record.error = String(err);
        record.durationMs = performance.now() - started;
        throw err;
      }
    };
  }
  const OriginalXHR = window.XMLHttpRequest;
  if (OriginalXHR) {
    window.XMLHttpRequest = function() {
      const xhr = new OriginalXHR();
      const record = { kind: 'xhr' };
      const open = xhr.open;
      xhr.open = function(method, url) { record.method = method; record.url = url; return open.apply(xhr, arguments); };
      xhr.addEventListener('loadend', () => { record.status = xhr.status; record.durationMs = performance.now() - record.startedPerf; });
      xhr.addEventListener('error', () => { record.error = 'XHR error'; });
      const send = xhr.send;
      xhr.send = function() { record.startedAt = Date.now(); record.startedPerf = performance.now(); state.network.push(record); return send.apply(xhr, arguments); };
      return xhr;
    };
  }
}
return true;
