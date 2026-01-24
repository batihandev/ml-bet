<script setup lang="ts">
import { reactive, ref, computed, h } from 'vue'
import type { TableColumn } from '@nuxt/ui'

const config = useRuntimeConfig()
const router = useRouter()

const form = reactive({
  startDate: '2024-12-30',
  endDate: '2025-06-30',
  edgeStart: 0.0,
  edgeEnd: 0.1,
  edgeStep: 0.01,
  evStart: 0.0,
  evEnd: 0.1,
  evStep: 0.01,
  min_bets: 300,
  bootstrap_n: 1000,
  stake: 1.0,
  kelly_mult: 0.0
})

const loading = ref(false)
const results = ref<any[]>([])
const sweepStats = ref<any>(null)

async function runSweep() {
  loading.value = true
  try {
    const res = await $fetch<any>(`${config.public.apiBase}/backtest/sweep`, {
      method: 'POST',
      body: {
        start_date: form.startDate,
        end_date: form.endDate,
        edge_range: [form.edgeStart, form.edgeEnd, form.edgeStep],
        ev_range: [form.evStart, form.evEnd, form.evStep],
        stake: form.stake,
        kelly_mult: form.kelly_mult,
        min_bets: form.min_bets,
        bootstrap_n: form.bootstrap_n
      }
    })
    results.value = res.cells || []
    sweepStats.value = res.summary || {}
  } catch (err) {
    console.error('Sweep error', err)
  } finally {
    loading.value = false
  }
}

async function fetchLatestSweep() {
  try {
    const res = await $fetch<any>(
      `${config.public.apiBase}/backtest/latest-sweep`
    )
    if (res && res.cells) {
      results.value = res.cells
      sweepStats.value = res.summary || {}

      // Update form to match latest results
      if (res.summary) {
        if (res.summary.start_date) form.startDate = res.summary.start_date
        if (res.summary.end_date) form.endDate = res.summary.end_date
        if (res.summary.edge_range) {
          form.edgeStart = res.summary.edge_range[0]
          form.edgeEnd = res.summary.edge_range[1]
          form.edgeStep = res.summary.edge_range[2]
        }
        if (res.summary.ev_range) {
          form.evStart = res.summary.ev_range[0]
          form.evEnd = res.summary.ev_range[1]
          form.evStep = res.summary.ev_range[2]
        }
        if (res.summary.stake !== undefined) form.stake = res.summary.stake
        if (res.summary.kelly_mult !== undefined)
          form.kelly_mult = res.summary.kelly_mult
        if (res.summary.min_bets !== undefined)
          form.min_bets = res.summary.min_bets
      }
    }
  } catch (err) {
    console.error('Failed to fetch latest sweep', err)
  }
}

onMounted(() => {
  fetchLatestSweep()
})

function runSingleBacktest(cell: any) {
  router.push({
    path: '/backtest',
    query: {
      startDate: form.startDate,
      endDate: form.endDate,
      minEdge: cell.min_edge,
      minEv: cell.min_ev,
      stake: form.stake,
      kellyMult: form.kelly_mult,
      autoRun: 'true'
    }
  })
}

const columns: TableColumn<any>[] = [
  { accessorKey: 'min_edge', header: 'Min Edge' },
  { accessorKey: 'min_ev', header: 'Min EV' },
  { accessorKey: 'bets', header: 'Bets' },
  {
    accessorKey: 'roi',
    header: 'ROI',
    cell: ({ row }) => {
      const v = (row.getValue('roi') as number) ?? 0
      const cls = v >= 0 ? 'text-emerald-500 font-medium' : 'text-red-500'
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  },
  {
    accessorKey: 'roi_p05',
    header: 'ROI (p05)',
    cell: ({ row }) => {
      const v = (row.getValue('roi_p05') as number) ?? null
      if (v === null)
        return h('span', { class: 'text-muted italic text-xs' }, 'low sample')
      const cls = v >= 0 ? 'text-emerald-600' : 'text-orange-500'
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  },
  {
    accessorKey: 'avg_odds',
    header: 'Avg Odds',
    cell: ({ row }) => (row.getValue('avg_odds') as number)?.toFixed(2)
  },
  {
    accessorKey: 'avg_ev',
    header: 'Avg EV',
    cell: ({ row }) => (row.getValue('avg_ev') as number)?.toFixed(3)
  },
  {
    id: 'actions',
    header: '',
    cell: ({ row }) => {
      return h(
        'UButton',
        {
          icon: 'i-lucide-external-link',
          size: 'xs',
          variant: 'ghost',
          onClick: () => runSingleBacktest(row.original)
        },
        'Run'
      )
    }
  }
]

// For the heatmap grid
const matrix = computed(() => {
  if (!results.value.length) return null
  const edges = [...new Set(results.value.map((c) => c.min_edge))].sort(
    (a, b) => a - b
  )
  const evs = [...new Set(results.value.map((c) => c.min_ev))].sort(
    (a, b) => b - a
  ) // top to bottom

  const grid = evs.map((ev) => {
    return {
      ev,
      cols: edges.map((edge) => {
        return results.value.find((c) => c.min_edge === edge && c.min_ev === ev)
      })
    }
  })
  return { edges, evs, grid }
})

function getHeatmapColor(roi: number, bets: number) {
  if (bets < form.min_bets) return 'bg-gray-100 dark:bg-gray-800 opacity-40'
  if (roi <= 0) return 'bg-red-50 dark:bg-red-950/20 text-red-600'
  if (roi < 0.05) return 'bg-emerald-50 dark:bg-emerald-950/20 text-emerald-600'
  if (roi < 0.1) return 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700'
  return 'bg-emerald-200 dark:bg-emerald-800/40 text-emerald-800 font-bold'
}
</script>

<template>
  <div class="space-y-6">
    <UBreadcrumb
      :items="[
        { label: 'Dashboard', to: '/' },
        { label: 'Backtests', to: '/backtest' },
        { label: 'Threshold Sweep' }
      ]"
    />

    <UPageHeader
      title="Threshold Sweep"
      description="Grid search over edge and EV thresholds to find the most stable betting parameters."
    />

    <UCard>
      <UForm :state="form" class="space-y-4">
        <div class="grid gap-4 md:grid-cols-4">
          <UFormField label="Start Date">
            <UInput v-model="form.startDate" type="date" />
          </UFormField>
          <UFormField label="End Date">
            <UInput v-model="form.endDate" type="date" />
          </UFormField>
          <UFormField label="Min Bets (Reliability)">
            <UInput v-model.number="form.min_bets" type="number" />
          </UFormField>
          <UFormField label="CI Samples (Bootstrap)">
            <UInput v-model.number="form.bootstrap_n" type="number" />
          </UFormField>
        </div>

        <div class="grid gap-4 md:grid-cols-2 border-t pt-4">
          <div class="space-y-2">
            <span class="text-xs font-medium text-muted"
              >Edge Range (Start, End, Step)</span
            >
            <div class="flex gap-2">
              <UInput
                v-model.number="form.edgeStart"
                type="number"
                step="0.01"
                class="w-full"
              />
              <UInput
                v-model.number="form.edgeEnd"
                type="number"
                step="0.01"
                class="w-full"
              />
              <UInput
                v-model.number="form.edgeStep"
                type="number"
                step="0.005"
                class="w-full"
              />
            </div>
          </div>
          <div class="space-y-2">
            <span class="text-xs font-medium text-muted"
              >EV Range (Start, End, Step)</span
            >
            <div class="flex gap-2">
              <UInput
                v-model.number="form.evStart"
                type="number"
                step="0.01"
                class="w-full"
              />
              <UInput
                v-model.number="form.evEnd"
                type="number"
                step="0.01"
                class="w-full"
              />
              <UInput
                v-model.number="form.evStep"
                type="number"
                step="0.005"
                class="w-full"
              />
            </div>
          </div>
        </div>

        <div class="flex justify-end pt-2">
          <UButton
            color="primary"
            icon="i-lucide-grid-3x3"
            :loading="loading"
            @click="runSweep"
          >
            Run Sweep
          </UButton>
        </div>
      </UForm>
    </UCard>

    <div v-if="matrix" class="grid gap-6 lg:grid-cols-2">
      <!-- Heatmap Grid -->
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold">Heatmap: ROI (%)</h3>
            <span class="text-[10px] text-muted">X=Edge, Y=EV</span>
          </div>
        </template>

        <div class="overflow-x-auto">
          <table class="w-full border-collapse text-[11px] min-w-[400px]">
            <thead>
              <tr>
                <th class="p-1 border bg-gray-50 dark:bg-gray-900">EV \ Eg</th>
                <th
                  v-for="e in matrix.edges"
                  :key="e"
                  class="p-1 border bg-gray-50 dark:bg-gray-900 w-12 text-center"
                >
                  {{ e }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in matrix.grid" :key="row.ev">
                <td
                  class="p-1 border font-medium bg-gray-50 dark:bg-gray-900 text-right"
                >
                  {{ row.ev }}
                </td>
                <td
                  v-for="(cell, idx) in row.cols"
                  :key="idx"
                  class="p-1 border text-center cursor-pointer hover:ring-2 ring-primary-500 ring-inset transition-all"
                  :class="getHeatmapColor(cell?.roi ?? 0, cell?.bets ?? 0)"
                  @click="cell && runSingleBacktest(cell)"
                >
                  <div v-if="cell">
                    {{ (cell.roi * 100).toFixed(0) }}%
                    <div class="text-[8px] opacity-70">{{ cell.bets }}</div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </UCard>

      <!-- Sortable Ranking -->
      <UCard>
        <template #header>
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-semibold">Top Performers</h3>
            <span class="text-[10px] text-muted">Sorted by ROI (p05)</span>
          </div>
        </template>

        <UTable
          :data="results"
          :columns="columns"
          class="w-full"
          :loading="loading"
        />
      </UCard>
    </div>

    <div
      v-else-if="!loading && results.length === 0"
      class="py-12 text-center text-muted border-2 border-dashed rounded-lg"
    >
      Adjust parameters and run sweep to find optimal betting zones.
    </div>
  </div>
</template>
