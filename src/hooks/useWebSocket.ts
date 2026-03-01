import { useEffect, useRef, useState, useCallback } from "react";

export type BackendAction = { type: string; [key: string]: unknown };
export type WsStatus = "connecting" | "connected" | "disconnected";

export function useWebSocket(
  url: string,
  onAction: (action: BackendAction) => void
): { send: (event: object) => void; wsStatus: WsStatus } {
  const [wsStatus, setWsStatus] = useState<WsStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onActionRef = useRef(onAction);
  onActionRef.current = onAction;

  useEffect(() => {
    if (!url) {
      setWsStatus("disconnected");
      return;
    }

    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setWsStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) { ws.close(); return; }
        setWsStatus("connected");
        retriesRef.current = 0;
      };

      ws.onmessage = (e) => {
        try {
          onActionRef.current(JSON.parse(e.data) as BackendAction);
        } catch {
          console.error("Bad WS message:", e.data);
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        if (retriesRef.current < 5) {
          const delay = Math.min(1000 * 2 ** retriesRef.current, 10000);
          retriesRef.current++;
          timerRef.current = setTimeout(connect, delay);
        } else {
          setWsStatus("disconnected");
        }
      };

      ws.onerror = () => console.error("WebSocket error");
    }

    connect();

    return () => {
      cancelled = true;
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [url]);

  const send = useCallback((event: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(event));
    }
  }, []);

  return { send, wsStatus };
}
