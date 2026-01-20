<script setup lang="ts">
const props = defineProps<{
  equity: any[]
}>()

const equityPath = computed(() => {
  if (!props.equity.length) return ''
  const margin = 20
  const width = 600
  const height = 160
  const data = props.equity.map((e) => e.cum_profit)
  const min = Math.min(0, ...data)
  const max = Math.max(0.1, ...data)
  const range = max - min

  return data
    .map((val, i) => {
      const x = margin + (i / (data.length - 1)) * (width - 2 * margin)
      const y = height - margin - ((val - min) / range) * (height - 2 * margin)
      return `${x},${y}`
    })
    .join(' ')
})
</script>

<template>
  <div class="h-40 w-full overflow-hidden">
    <svg
      v-if="equity.length"
      width="100%"
      height="100%"
      viewBox="0 0 600 160"
      preserveAspectRatio="none"
    >
      <!-- Grid line for zero -->
      <line
        x1="0"
        y1="140"
        x2="600"
        y2="140"
        stroke="currentColor"
        stroke-dasharray="2 4"
        class="text-gray-200 dark:text-gray-800"
      />
      <!-- Path -->
      <polyline
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        :points="equityPath"
        class="text-primary-500"
      />
    </svg>
    <div
      v-else
      class="flex h-40 items-center justify-center rounded-lg border border-dashed border-gray-300/60 text-[11px] text-muted dark:border-gray-700/80"
    >
      No data available to show equity curve.
    </div>
  </div>
</template>
