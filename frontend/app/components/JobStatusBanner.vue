<script setup lang="ts">
import { computed, ref, watch, onBeforeUnmount } from 'vue'

const { isProcessing, lastEvent } = useJobEvents()

const startedAt = ref<number | null>(null)
const elapsed = ref<string>('')
const showAfterDone = ref(false)
let timer: ReturnType<typeof setInterval> | null = null
let hideTimer: ReturnType<typeof setTimeout> | null = null

const jobLabels: Record<string, string> = {
  backtest: 'Backtest',
  sweep: 'Sweep',
  training: 'Training',
  dataset_build: 'Dataset build',
  features_build: 'Features build',
  unzip: 'Unzip'
}

function parseEventType(type?: string | null) {
  if (!type) return { key: null, status: null }
  const match = type.match(/^(.*)_(started|completed|failed|progress)$/)
  if (!match) return { key: type, status: null }
  return { key: match[1], status: match[2] }
}

function formatElapsed(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function updateElapsed() {
  if (!startedAt.value) {
    elapsed.value = ''
    return
  }
  elapsed.value = formatElapsed(Date.now() - startedAt.value)
}

watch(
  () => lastEvent.value,
  (val) => {
    const { status } = parseEventType(val?.type)
    if (status === 'started') {
      startedAt.value = Date.now()
      updateElapsed()
      if (!timer) timer = setInterval(updateElapsed, 1000)
      showAfterDone.value = false
      if (hideTimer) {
        clearTimeout(hideTimer)
        hideTimer = null
      }
    } else if (status === 'completed' || status === 'failed') {
      updateElapsed()
      if (timer) {
        clearInterval(timer)
        timer = null
      }
      showAfterDone.value = true
      if (hideTimer) clearTimeout(hideTimer)
      hideTimer = setTimeout(() => {
        showAfterDone.value = false
      }, 12000)
    }
  }
)

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  if (hideTimer) clearTimeout(hideTimer)
})

const eventMeta = computed(() => parseEventType(lastEvent.value?.type))
const jobLabel = computed(() => {
  const key = eventMeta.value.key
  if (!key) return 'Job'
  return jobLabels[key] || key.replace(/_/g, ' ')
})

const statusText = computed(() => {
  const status = eventMeta.value.status
  if (status === 'failed') return `${jobLabel.value} failed`
  if (status === 'completed') return `${jobLabel.value} completed`
  if (status === 'started' || isProcessing.value)
    return `Running ${jobLabel.value}…`
  if (status === 'progress') return `Running ${jobLabel.value}…`
  return `${jobLabel.value} status`
})

const detailText = computed(() => {
  const payload = lastEvent.value?.payload
  const key = eventMeta.value.key
  if (!payload || !key) return ''

  if (key === 'backtest') {
    const params = payload.params || payload
    const parts = []
    if (params?.min_edge !== undefined) parts.push(`edge ≥ ${params.min_edge}`)
    if (params?.min_ev !== undefined) parts.push(`ev ≥ ${params.min_ev}`)
    if (params?.selection_mode) parts.push(params.selection_mode)
    if (params?.start_date && params?.end_date) {
      parts.push(`${params.start_date} → ${params.end_date}`)
    }
    return parts.join(' • ')
  }

  if (key === 'sweep') {
    const params = payload.params || payload
    const parts = []
    if (params?.edge_range) parts.push(`edge ${params.edge_range.join('..')}`)
    if (params?.ev_range) parts.push(`ev ${params.ev_range.join('..')}`)
    if (params?.selection_mode) parts.push(params.selection_mode)
    if (parts.length) return parts.join(' • ')
    if (params?.done && params?.total) {
      const cell = `cell ${params.done}/${params.total}`
      const gate =
        params?.min_edge !== undefined && params?.min_ev !== undefined
          ? `edge ${params.min_edge}, ev ${params.min_ev}`
          : ''
      return [cell, gate].filter(Boolean).join(' • ')
    }
  }

  if (key === 'training') {
    const params = payload.params || payload
    if (params?.train_start && params?.training_cutoff_date) {
      return `${params.train_start} → ${params.training_cutoff_date}`
    }
  }

  if (key === 'dataset_build') return 'Matches.csv → matches.csv'
  if (key === 'features_build') return 'matches.csv → features.csv'
  if (key === 'unzip') return 'Extracting raw dataset'

  return ''
})

const statusClass = computed(() => {
  const status = eventMeta.value.status
  if (status === 'failed') return 'text-red-600'
  if (status === 'completed') return 'text-emerald-600'
  return 'text-primary-900 dark:text-primary-50'
})

const statusIcon = computed(() => {
  const status = eventMeta.value.status
  if (status === 'failed') return 'i-lucide-x-circle'
  if (status === 'completed') return 'i-lucide-check-circle'
  return 'i-lucide-activity'
})

const progressText = computed(() => {
  if (eventMeta.value.status !== 'progress') return ''
  const payload: any = lastEvent.value?.payload
  const pct = payload?.pct
  if (typeof pct === 'number') return `${(pct * 100).toFixed(0)}%`
  return ''
})

const showBanner = computed(() => isProcessing.value || showAfterDone.value)
</script>

<template>
  <Transition name="fade">
    <div
      v-if="showBanner"
      class="border-b border-gray-200/70 bg-primary-50/80 backdrop-blur-sm text-primary-900 dark:border-gray-800 dark:bg-primary-900/40 dark:text-primary-50"
    >
      <div
        class="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-2 text-xs"
      >
        <div class="flex items-center gap-2">
          <UIcon
            :name="statusIcon"
            class="h-4 w-4"
            :class="isProcessing ? 'animate-pulse' : ''"
          />
          <span class="font-medium" :class="statusClass">
            {{ statusText }}
          </span>
          <span v-if="progressText" class="text-[11px] text-muted">
            • {{ progressText }}
          </span>
          <span v-if="detailText" class="text-[11px] text-muted">
            • {{ detailText }}
          </span>
        </div>

        <div class="flex items-center gap-3">
          <span
            v-if="isProcessing && elapsed"
            class="rounded-full bg-white/60 px-2 py-0.5 text-[10px] text-gray-700 dark:bg-gray-900/40 dark:text-gray-200"
          >
            Elapsed {{ elapsed }}
          </span>
          <span class="hidden text-[10px] opacity-70 sm:inline">
            Live status via WebSocket
          </span>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease-out, transform 0.15s ease-out;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
