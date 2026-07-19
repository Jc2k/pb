import { useEffect, useRef, useState } from "react";
import { PageShell } from "../components/PageShell";

export interface WebSettings {
  prevent_sleep_while_working: boolean;
  prevent_sleep_supported: boolean;
  prevent_sleep_active: boolean;
  prevent_sleep_error?: string;
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

export function SettingsPage() {
  const [settings, setSettings] = useState<WebSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const requestEpoch = useRef(0);
  const savingRef = useRef(false);

  const refreshSettings = async () => {
    const epoch = requestEpoch.current;
    const response = await fetch("/api/settings");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
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
          prevent_sleep_while_working:
            !settings.prevent_sleep_while_working,
        }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
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

  return (
    <PageShell contentClassName="settings-page-wrap">
      <section className="hero-section project-settings-hero settings-hero">
        <h1>Settings</h1>
        <p>Personal preferences for this pb service.</p>
      </section>

      {settings
        ? (
          <div className="project-settings-stack">
            <SleepSettingControl
              settings={settings}
              saving={saving}
              error={error}
              onToggle={() => void togglePreventSleep()}
            />
          </div>
        )
        : (
          <section className="settings-card settings-loading" role="status">
            {error ? `Could not load settings: ${error}` : "Loading settings…"}
          </section>
        )}
    </PageShell>
  );
}
