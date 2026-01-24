<script setup lang="ts">
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale
} from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale)

const props = defineProps<{
  data: { feature: string; importance: number }[]
  loading?: boolean
}>()

const chartData = computed(() => ({
  labels: props.data.map((d) => d.feature),
  datasets: [
    {
      label: 'Feature Importance',
      backgroundColor: '#10b981', // Tailwind emerald-500
      data: props.data.map((d) => d.importance)
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  indexAxis: 'y' as const,
  plugins: {
    legend: {
      display: false
    },
    tooltip: {
      callbacks: {
        label: (context: any) => {
          return `Importance: ${context.parsed.x.toFixed(4)}`
        }
      }
    }
  },
  scales: {
    x: {
      beginAtZero: true,
      grid: {
        display: false
      }
    },
    y: {
      grid: {
        display: false
      }
    }
  }
}
</script>

<template>
  <div class="space-y-4">
    <div v-if="loading" class="flex h-64 items-center justify-center">
      <UIcon
        name="i-lucide-loader-2"
        class="h-8 w-8 animate-spin text-primary"
      />
    </div>
    <div v-else-if="data.length" class="h-96 w-full">
      <Bar :data="chartData" :options="chartOptions" />
    </div>
    <div
      v-else
      class="flex h-64 items-center justify-center rounded-lg border border-dashed border-gray-300/60 text-sm text-muted dark:border-gray-700/80"
    >
      No feature importance data. Train the model to see insights.
    </div>
  </div>
</template>
