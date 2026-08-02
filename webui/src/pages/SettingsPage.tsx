import { useCallback, useEffect, useRef, useState } from "react";
import { PageShell } from "../components/PageShell";
import { apiErrorMessage } from "../lib/integrationConfig";

export interface WebSettings {
  prevent_sleep_while_working: boolean;
  prevent_sleep_supported: boolean;
  prevent_sleep_active: boolean;
  prevent_sleep_error?: string;
}

export type TailscaleState =
  | "unavailable"
  | "disconnected"
  | "available"
  | "needs_repair"
  | "authorization_required"
  | "conflict"
  | "active"
  | "error";

export interface TailscaleSettings {
  state: TailscaleState;
  installed: boolean;
  connected: boolean;
  enabled: boolean;
  active: boolean;
  https_port: number;
  backend_target: string;
  url?: string;
  authorization_url?: string;
  error?: string;
  direct_lan_access: boolean;
}

export function SleepSettingControl(
  {
    settings,
    saving = false,
    error = "",
    onToggle,
  }: {
    settings: WebSettings;
    saving?: boolean;
    error?: string;
    onToggle?: () => void;
  },
) {
  const enabled = settings.prevent_sleep_while_working;
  const supported = settings.prevent_sleep_supported;
  const detail = !supported
    ? "This idle-sleep assertion is available on macOS only."
    : settings.prevent_sleep_active
    ? "Active now — this Mac will stay awake while pb works."
    : enabled
    ? "Ready — pb will keep this Mac awake whenever queued work is running."
    : "Off — macOS can sleep normally while pb works.";

  return (
    <section
      className="settings-card notification-card power-settings-card"
      aria-labelledby="prevent-sleep-title"
    >
      <div>
        <h2 id="prevent-sleep-title">Prevent sleep while working</h2>
        <p>
          Hold an idle system-sleep assertion while the work queue is being
          processed. The display may still turn off.
        </p>
        <p
          className={`power-setting-status ${
            settings.prevent_sleep_active ? "is-active" : ""
          }`}
          role="status"
        >
          {detail}
        </p>
        {(error || settings.prevent_sleep_error) && (
          <p className="power-setting-error" role="alert">
            {error || settings.prevent_sleep_error}
          </p>
        )}
      </div>
      <button
        type="button"
        className={`notification-switch ${enabled ? "is-on" : ""}`}
        role="switch"
        aria-checked={enabled}
        disabled={!supported || saving}
        onClick={onToggle}
      >
        <span>{saving ? "Saving…" : enabled ? "On" : "Off"}</span>
        <i></i>
      </button>
    </section>
  );
}

function tailscaleCopy(status: TailscaleSettings) {
  switch (status.state) {
    case "unavailable":
      return {
        label: "Not installed",
        detail:
          "Install Tailscale on this Mac to publish pb securely to your tailnet.",
      };
    case "disconnected":
      return {
        label: "Not connected",
        detail: "Connect this Mac to a tailnet, then refresh the connection.",
      };
    case "available":
      return {
        label: "Ready",
        detail:
          "Tailscale is connected. pb can create and maintain a private HTTPS address.",
      };
    case "needs_repair":
      return {
        label: "Needs repair",
        detail:
          "Secure access is enabled, but pb’s Tailscale endpoint is missing.",
      };
    case "authorization_required":
      return {
        label: "Approval needed",
        detail:
          "Your tailnet requires an administrator to approve Tailscale Serve once.",
      };
    case "conflict":
      return {
        label: "Port in use",
        detail:
          `Tailscale HTTPS port ${status.https_port} already belongs to another Serve endpoint. pb left it unchanged.`,
      };
    case "active":
      return {
        label: "Secure access active",
        detail: status.enabled
          ? "pb will keep this private HTTPS endpoint available after restarts."
          : "A matching endpoint already exists. Let pb manage it to keep it available after restarts.",
      };
    case "error":
      return {
        label: "Could not inspect Tailscale",
        detail:
          "pb could not read the local Tailscale state. Your existing configuration was not changed.",
      };
  }
}

export function TailscaleAccessControl(
  {
    status,
    busy = false,
    error = "",
    onSetEnabled,
    onRefresh,
  }: {
    status: TailscaleSettings;
    busy?: boolean;
    error?: string;
    onSetEnabled?: (enabled: boolean) => void;
    onRefresh?: () => void;
  },
) {
  const [copied, setCopied] = useState(false);
  const copy = tailscaleCopy(status);
  const displayedError = error || status.error || "";

  const copyUrl = async () => {
    if (!status.url) return;
    try {
      await navigator.clipboard.writeText(status.url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1_500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section
      className={`settings-card tailscale-settings-card is-${status.state}`}
      aria-labelledby="tailscale-access-title"
    >
      <div className="tailscale-settings-heading">
        <div>
          <h2 id="tailscale-access-title">Secure remote access</h2>
          <p>
            Let pb manage a tailnet-only HTTPS address for Safari, voice input,
            and remote use.
          </p>
        </div>
        <span className="tailscale-state-badge" role="status">
          {copy.label}
        </span>
      </div>

      <p className="tailscale-setting-detail">{copy.detail}</p>

      {status.url && (
        <a className="tailscale-address" href={status.url}>
          {status.url}
        </a>
      )}

      {status.active && status.direct_lan_access && (
        <p className="tailscale-lan-warning">
          pb is also listening for direct LAN HTTP connections. Use the HTTPS
          address above for voice input; set <code>web.listen</code> to{" "}
          <code>127.0.0.1</code> if Tailscale should be the only remote path.
        </p>
      )}

      {displayedError && (
        <p className="power-setting-error" role="alert">
          {displayedError}
        </p>
      )}

      <div className="tailscale-settings-actions">
        {status.state === "unavailable" && (
          <a
            className="btn btn-outline-primary"
            href="https://tailscale.com/download/mac"
            target="_blank"
            rel="noreferrer"
          >
            Get Tailscale
          </a>
        )}
        {status.state === "disconnected" && status.authorization_url && (
          <a
            className="btn btn-primary"
            href={status.authorization_url}
            target="_blank"
            rel="noreferrer"
          >
            Sign in to Tailscale
          </a>
        )}
        {status.state === "authorization_required" &&
          status.authorization_url && (
          <a
            className="btn btn-primary"
            href={status.authorization_url}
            target="_blank"
            rel="noreferrer"
          >
            Approve secure access
          </a>
        )}
        {status.state === "authorization_required" && (
          <button
            type="button"
            className="btn btn-outline-primary"
            disabled={busy}
            onClick={() => onSetEnabled?.(true)}
          >
            {busy ? "Checking…" : "Retry setup"}
          </button>
        )}
        {(status.state === "available" ||
          status.state === "needs_repair") && (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => onSetEnabled?.(true)}
          >
            {busy
              ? "Working…"
              : status.state === "needs_repair"
              ? "Repair secure access"
              : "Enable secure access"}
          </button>
        )}
        {status.state === "active" && status.url && (
          <>
            <a className="btn btn-primary" href={status.url}>Open pb</a>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => void copyUrl()}
            >
              {copied ? "Copied" : "Copy address"}
            </button>
          </>
        )}
        {status.state === "active" && (
          <button
            type="button"
            className={status.enabled
              ? "btn btn-outline-danger"
              : "btn btn-outline-primary"}
            disabled={busy}
            onClick={() => onSetEnabled?.(!status.enabled)}
          >
            {busy
              ? "Working…"
              : status.enabled
              ? "Disable secure access"
              : "Manage with pb"}
          </button>
        )}
        {(status.state === "unavailable" ||
          status.state === "disconnected" ||
          status.state === "conflict" ||
          status.state === "error") && (
          <button
            type="button"
            className="btn btn-outline-secondary"
            disabled={busy}
            onClick={onRefresh}
          >
            {busy ? "Checking…" : "Refresh"}
          </button>
        )}
      </div>
    </section>
  );
}

export function SettingsPage() {
  const [settings, setSettings] = useState<WebSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const requestEpoch = useRef(0);
  const savingRef = useRef(false);
  const [tailscale, setTailscale] = useState<TailscaleSettings | null>(null);
  const [tailscaleBusy, setTailscaleBusy] = useState(false);
  const [tailscaleError, setTailscaleError] = useState("");
  const tailscaleRequestEpoch = useRef(0);
  const tailscaleBusyRef = useRef(false);
  const tailscaleRefreshingRef = useRef(false);

  const refreshSettings = async () => {
    const epoch = requestEpoch.current;
    const response = await fetch("/api/settings");
    if (!response.ok) {
      throw new Error(
        await apiErrorMessage(response, "Could not load settings"),
      );
    }
    const next = (await response.json()) as WebSettings;
    if (epoch === requestEpoch.current) setSettings(next);
  };

  useEffect(() => {
    let mounted = true;
    const refresh = async () => {
      if (savingRef.current) return;
      const epoch = requestEpoch.current;
      try {
        const response = await fetch("/api/settings");
        if (!response.ok) {
          throw new Error(
            await apiErrorMessage(response, "Could not load settings"),
          );
        }
        const next = (await response.json()) as WebSettings;
        if (mounted && epoch === requestEpoch.current) {
          setSettings(next);
          setError("");
        }
      } catch (requestError) {
        if (mounted && epoch === requestEpoch.current) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Could not load settings",
          );
        }
      }
    };

    void refresh();
    const interval = window.setInterval(() => void refresh(), 3_000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const refreshTailscale = useCallback(async () => {
    if (tailscaleBusyRef.current || tailscaleRefreshingRef.current) return;
    tailscaleRefreshingRef.current = true;
    const epoch = tailscaleRequestEpoch.current;
    try {
      const response = await fetch("/api/settings/tailscale");
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Could not inspect Tailscale"),
        );
      }
      const next = (await response.json()) as TailscaleSettings;
      if (epoch === tailscaleRequestEpoch.current) {
        setTailscale(next);
        setTailscaleError("");
      }
    } catch (requestError) {
      if (epoch === tailscaleRequestEpoch.current) {
        setTailscaleError(
          requestError instanceof Error
            ? requestError.message
            : "Could not inspect Tailscale",
        );
      }
    } finally {
      tailscaleRefreshingRef.current = false;
    }
  }, []);

  useEffect(() => {
    void refreshTailscale();
    const interval = window.setInterval(
      () => void refreshTailscale(),
      15_000,
    );
    return () => {
      tailscaleRequestEpoch.current += 1;
      window.clearInterval(interval);
    };
  }, [refreshTailscale]);

  const togglePreventSleep = async () => {
    if (!settings || saving) return;
    requestEpoch.current += 1;
    savingRef.current = true;
    setSaving(true);
    setError("");
    try {
      const response = await fetch("/api/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prevent_sleep_while_working: !settings.prevent_sleep_while_working,
        }),
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Could not save setting"),
        );
      }
      setSettings((await response.json()) as WebSettings);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not save setting",
      );
      void refreshSettings().catch(() => {});
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };

  const setTailscaleEnabled = async (enabled: boolean) => {
    if (tailscaleBusy) return;
    tailscaleRequestEpoch.current += 1;
    tailscaleBusyRef.current = true;
    setTailscaleBusy(true);
    setTailscaleError("");
    try {
      const response = await fetch("/api/settings/tailscale", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Could not change secure access"),
        );
      }
      setTailscale((await response.json()) as TailscaleSettings);
    } catch (requestError) {
      setTailscaleError(
        requestError instanceof Error
          ? requestError.message
          : "Could not change secure access",
      );
      tailscaleBusyRef.current = false;
      await refreshTailscale();
    } finally {
      tailscaleBusyRef.current = false;
      setTailscaleBusy(false);
    }
  };

  return (
    <PageShell contentClassName="settings-page-wrap">
      <section className="hero-section project-settings-hero settings-hero">
        <h1>Settings</h1>
        <p>Personal preferences for this pb service.</p>
      </section>

      <div className="project-settings-stack">
        {tailscale
          ? (
            <TailscaleAccessControl
              status={tailscale}
              busy={tailscaleBusy}
              error={tailscaleError}
              onSetEnabled={(enabled) => void setTailscaleEnabled(enabled)}
              onRefresh={() => void refreshTailscale()}
            />
          )
          : (
            <section className="settings-card settings-loading" role="status">
              {tailscaleError
                ? `Could not inspect Tailscale: ${tailscaleError}`
                : "Checking secure remote access…"}
            </section>
          )}

        {settings
          ? (
            <SleepSettingControl
              settings={settings}
              saving={saving}
              error={error}
              onToggle={() => void togglePreventSleep()}
            />
          )
          : (
            <section className="settings-card settings-loading" role="status">
              {error
                ? `Could not load settings: ${error}`
                : "Loading settings…"}
            </section>
          )}
      </div>
    </PageShell>
  );
}
