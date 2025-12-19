// composables/useJobEvents.ts
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRuntimeConfig, useToast } from '#imports'

type JobEvent =
  | { type: 'backtest_started'; payload: any }
  | { type: 'backtest_completed'; payload: any }
  | { type: 'backtest_failed'; payload: any }
  | { type: string; payload: any }

export function useJobEvents() {
  const isProcessing = ref(false)
  const lastEvent = ref<JobEvent | null>(null)
  const socket = ref<WebSocket | null>(null)

  const config = useRuntimeConfig()
  const toast = useToast()

  const connect = () => {
    if (socket.value) return

    const apiBase = config.public.apiBase as string
    const wsUrl = apiBase.replace(/^http/, 'ws') + '/ws/progress'

    const ws = new WebSocket(wsUrl)
    socket.value = ws

    ws.onopen = () => {
      // Optionally send a ping to keep server loop alive
      ws.send('hello')
    }

    ws.onmessage = (event) => {
      try {
        const msg: JobEvent = JSON.parse(event.data)
        lastEvent.value = msg

        if (msg.type === 'backtest_started') {
          isProcessing.value = true
        } else if (msg.type === 'backtest_completed') {
          isProcessing.value = false

          const summary = msg.payload?.summary ?? {}
          const roi =
            typeof summary.overall_roi === 'number' ? summary.overall_roi : null

          toast.add({
            title: 'Backtest finished',
            description:
              roi !== null
                ? `Overall ROI: ${(roi * 100).toFixed(2)}%`
                : 'Backtest completed successfully.',
            color: roi !== null && roi > 0 ? 'success' : 'info'
          })
        } else if (msg.type === 'backtest_failed') {
          isProcessing.value = false
          toast.add({
            title: 'Backtest failed',
            description: msg.payload?.error ?? 'Check backend logs.',
            color: 'error'
          })
        }
      } catch (err) {
        console.error('Failed to parse WS message', err)
      }
    }

    ws.onclose = () => {
      socket.value = null
      isProcessing.value = false
    }

    ws.onerror = () => {
      // Fail silently for now; you can add a toast if you want.
    }
  }

  onMounted(connect)

  onBeforeUnmount(() => {
    if (socket.value) {
      socket.value.close()
      socket.value = null
    }
  })

  return {
    isProcessing,
    lastEvent
  }
}
