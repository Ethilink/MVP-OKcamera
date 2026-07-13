import { Button } from "@/components/ui/button"

type StartProps = {
  mode: "start"
  /** Gate result (contract §/status). When false, `reason` says why. */
  enabled: boolean
  reason?: string
  pending?: boolean
  onStart: () => void
}

type StopProps = {
  mode: "stop"
  pending?: boolean
  onStop: () => void
}

/**
 * The one primary action on the live screen. In `setup`/`finished` it is the
 * gated **Start** (disabled Start always states why — the operator needs to know
 * it is waiting, not broken). While `recording` it is **Stop**, always enabled.
 */
export function StartStopControl(props: StartProps | StopProps) {
  if (props.mode === "stop") {
    return (
      <Button
        size="lg"
        variant="destructive"
        disabled={props.pending}
        onClick={props.onStop}
      >
        Stop
      </Button>
    )
  }

  return (
    <div className="flex flex-col items-center gap-1.5">
      <Button
        size="lg"
        disabled={!props.enabled || props.pending}
        onClick={props.onStart}
      >
        Start
      </Button>
      {!props.enabled && props.reason && (
        <p className="text-xs text-muted-foreground">{props.reason}</p>
      )}
    </div>
  )
}
