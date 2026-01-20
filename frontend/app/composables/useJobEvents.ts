// composables/useJobEvents.ts
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRuntimeConfig, useToast } from '#imports'

type JobEvent =
  | { type: 'backtest_started'; payload: any }
  | { type: 'backtest_completed'; payload: any }
  | { type: 'backtest_failed'; payload: any }
  | { type: string; payload: any }

// shared state
const isProcessing = ref(false)
const lastEvent = ref<JobEvent | null>(null)
const socket = ref<WebSocket | null>(null)

export function useJobEvents() {
  const config = useRuntimeConfig()
  const toast = useToast()

  const connect = () => {
    if (socket.value || import.meta.server) return

    const apiBase = config.public.apiBase as string
    const wsUrl = apiBase.replace(/^http/, 'ws') + '/ws/progress'

    const ws = new WebSocket(wsUrl)
    socket.value = ws

    ws.onopen = () => {
      ws.send('hello')
    }

    ws.onmessage = (event) => {
      try {
        const msg: JobEvent = JSON.parse(event.data)
        lastEvent.value = msg

        if (
          msg.type === 'backtest_started' ||
          msg.type === 'training_started' ||
          msg.type === 'dataset_build_started' ||
          msg.type === 'features_build_started' ||
          msg.type === 'unzip_started'
        ) {
          isProcessing.value = true
          if (msg.type === 'training_started') {
            toast.add({
              title: 'Training started',
              description: 'The model training job has been dispatched.',
              color: 'info'
            })
          } else if (msg.type === 'dataset_build_started') {
            toast.add({
              title: 'Dataset build started',
              description: 'Processing raw Matches.csv...',
              color: 'info'
            })
          } else if (msg.type === 'features_build_started') {
            toast.add({
              title: 'Features build started',
              description: 'Engineering features from dataset...',
              color: 'info'
            })
          } else if (msg.type === 'unzip_started') {
            toast.add({
              title: 'Extraction started',
              description: 'Unzipping raw match data...',
              color: 'info'
            })
          }
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
        } else if (msg.type === 'training_completed') {
          isProcessing.value = false
          toast.add({
            title: 'Training finished',
            description: 'All models have been retrained successfully.',
            color: 'success'
          })
        } else if (msg.type === 'dataset_build_completed') {
          isProcessing.value = false
          toast.add({
            title: 'Dataset ready',
            description: 'matches.csv has been built.',
            color: 'success'
          })
        } else if (msg.type === 'features_build_completed') {
          isProcessing.value = false
          toast.add({
            title: 'Features ready',
            description: 'features.csv has been built.',
            color: 'success'
          })
        } else if (msg.type === 'unzip_completed') {
          isProcessing.value = false
          toast.add({
            title: 'Extraction ready',
            description: 'Raw data extracted to data/raw/.',
            color: 'success'
          })
        } else if (
          msg.type === 'backtest_failed' ||
          msg.type === 'training_failed' ||
          msg.type === 'dataset_build_failed' ||
          msg.type === 'features_build_failed' ||
          msg.type === 'unzip_failed'
        ) {
          isProcessing.value = false
          toast.add({
            title: 'Job failed',
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
  }

  if (import.meta.client) {
    onMounted(connect)
  }

  return {
    isProcessing,
    lastEvent
  }
}
