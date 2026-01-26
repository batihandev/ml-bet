<script setup lang="ts">
import { reactive, ref, computed, h, resolveComponent } from 'vue'
import type { TableColumn } from '@nuxt/ui'

const config = useRuntimeConfig()
const router = useRouter()

const UButton = resolveComponent('UButton')

const form = reactive({
  startDate: '2024-12-30',
  endDate: '2025-06-30',
  edgeStart: 0.0,
  edgeEnd: 0.1,
  edgeStep: 0.01,
  evStart: 0.0,
  evEnd: 0.1,
  evStep: 0.01,
  alphaStart: 0.0,
  alphaEnd: 1.0,
  alphaStep: 0.1,
  min_bets: 300,
  bootstrap_n: 1000,
  stake: 1.0,
  kelly_mult: 0.0,
  selectionMode: 'best_ev'
})

const loading = ref(false)
const results = ref<any[]>([])
const sweepStats = ref<any>(null)
const alphaResults = ref<Record<string, any>>({})
const selectedAlpha = ref<string>('')
const showAllRows = ref(false)
const detailsOpen = ref(false)
const selectedCell = ref<any | null>(null)

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
        alpha_range: [form.alphaStart, form.alphaEnd, form.alphaStep],
        stake: form.stake,
        kelly_mult: form.kelly_mult,
        min_bets: form.min_bets,
        bootstrap_n: form.bootstrap_n,
        selection_mode: form.selectionMode,
        debug: 0
      }
    })
    loadSweepResults(res)
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
    if (res && (res.cells || res.alpha_results)) {
      loadSweepResults(res)
      // Form values are preserved as defaults, results are loaded separately
    }
  } catch (err) {
    console.error('Failed to fetch latest sweep', err)
  }
}

onMounted(() => {
  fetchLatestSweep()
})

function runSingleBacktest(cell: any) {
  const alpha =
    cell?.alpha !== undefined && cell?.alpha !== null
      ? parseFloat(String(cell.alpha))
      : selectedAlpha.value
        ? parseFloat(selectedAlpha.value)
        : (sweepStats.value?.blend_alpha ?? 1)
  const startDate = sweepStats.value?.start_date || form.startDate
  const endDate = sweepStats.value?.end_date || form.endDate
  router.push({
    path: '/backtest',
    query: {
      startDate,
      endDate,
      minEdge: cell.min_edge,
      minEv: cell.min_ev,
      stake: form.stake,
      kellyMult: form.kelly_mult,
      selectionMode: form.selectionMode,
      blendAlpha: alpha,
      autoRun: 'true'
    }
  })
}

function openDetails(cell: any) {
  selectedCell.value = cell
  detailsOpen.value = true
}

const getTopPasses = (cell: any) =>
  cell?.n_top_prob_passes_gate ?? cell?.n_top_choice_value ?? 0
const getAnyPasses = (cell: any) =>
  cell?.n_any_passes_gate ?? cell?.n_any_outcome_value ?? 0

const formatPercent = (num?: number, denom?: number) => {
  if (!denom || !Number.isFinite(num) || !Number.isFinite(denom))
    return '–'
  return `${((num / denom) * 100).toFixed(1)}%`
}

const formatMix = (mix?: { home?: number; draw?: number; away?: number }) => {
  if (!mix) return '–'
  const { home, draw, away } = mix
  if (
    !Number.isFinite(home) ||
    !Number.isFinite(draw) ||
    !Number.isFinite(away)
  ) {
    return '–'
  }
  return `${(home * 100).toFixed(1)}% H / ${(draw * 100).toFixed(1)}% D / ${(away * 100).toFixed(1)}% A`
}

const allCells = computed(() => {
  const alphas = Object.keys(alphaResults.value || {})
  if (alphas.length) {
    return alphas.flatMap((alpha) => {
      const payload = alphaResults.value?.[alpha]
      const cells = payload?.cells || []
      return cells.map((cell: any) => ({ ...cell, alpha }))
    })
  }
  const fallbackAlpha =
    selectedAlpha.value || (sweepStats.value?.blend_alpha ?? '1')
  return results.value.map((cell) => ({ ...cell, alpha: fallbackAlpha }))
})

const sortedAllCells = computed(() => {
  return [...allCells.value].sort((a, b) => {
    const ap = a?.roi_p05 ?? -999
    const bp = b?.roi_p05 ?? -999
    if (bp !== ap) return bp - ap
    return (b?.roi ?? 0) - (a?.roi ?? 0)
  })
})

const topResults = computed(() => {
  const src = sortedAllCells.value
  if (showAllRows.value) return src
  return src.slice(0, 8)
})

const alphaOptions = computed(() => {
  const keys = Object.keys(alphaResults.value || {})
  const sorted = keys.sort((a, b) => parseFloat(a) - parseFloat(b))
  return sorted.map((k) => ({ label: k, value: k }))
})

function loadSweepResults(res: any) {
  if (res?.alpha_results) {
    alphaResults.value = res.alpha_results || {}
    const preferred =
      res.default_alpha ||
      (alphaResults.value['1'] ? '1' : Object.keys(alphaResults.value)[0] || '')
    if (preferred) {
      applyAlpha(preferred)
    }
  } else {
    alphaResults.value = {}
    selectedAlpha.value = ''
    results.value = res.cells || []
    sweepStats.value = res.summary || {}
  }
}

function applyAlpha(alpha: string) {
  const payload = alphaResults.value?.[alpha]
  if (!payload) return
  selectedAlpha.value = alpha
  results.value = payload.cells || []
  sweepStats.value = payload.summary || {}
}

function selectAlpha(alpha: string) {
  applyAlpha(alpha)
}

const columns: TableColumn<any>[] = [
  {
    id: 'details',
    header: '',
    cell: ({ row }) =>
      h(UButton, {
        icon: 'i-lucide-info',
        size: 'sm',
        variant: 'subtle',
        color: 'info',
        label: 'Details',
        onClick: () => openDetails(row.original)
      })
  },
  { accessorKey: 'min_edge', header: 'Min Edge' },
  { accessorKey: 'min_ev', header: 'Min EV' },
  {
    accessorKey: 'alpha',
    header: 'Alpha',
    cell: ({ row }) => {
      const v = row.original.alpha
      if (v === undefined || v === null) return '–'
      return Number.isFinite(parseFloat(String(v)))
        ? parseFloat(String(v)).toFixed(2).replace(/\.00$/, '')
        : String(v)
    }
  },
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
    accessorKey: 'pass_rate_top',
    header: 'Pass Rate (Top)',
    cell: ({ row }) => {
      const top = getTopPasses(row.original)
      const total = row.original.n_all_valid || 0
      return formatPercent(top, total)
    }
  },
  {
    accessorKey: 'avg_ev',
    header: 'Avg EV',
    cell: ({ row }) => (row.getValue('avg_ev') as number)?.toFixed(3)
  },
  {
    id: 'run',
    header: '',
    cell: ({ row }) =>
      h(UButton, {
        icon: 'i-lucide-external-link',
        size: 'sm',
        variant: 'solid',
        color: 'success',
        label: 'Run',
        onClick: () => runSingleBacktest(row.original)
      })
  }
]

// For the heatmap grid
const matrix = computed(() => {
  if (!results.value.length) return null
  const edges = [...new Set(results.value.map((c) => c.min_edge))].sort(
    (a, b) => a - b
  )
  const evs = [...new Set(results.value.map((c) => c.min_ev))].sort(
    (a, b) => a - b
  ) // start from 0

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
          <div class="space-y-2">
            <span class="text-xs font-medium text-muted"
              >Alpha Range (Start, End, Step)</span
            >
            <div class="flex gap-2">
              <UInput
                v-model.number="form.alphaStart"
                type="number"
                step="0.05"
                min="0"
                max="1"
                class="w-full"
              />
              <UInput
                v-model.number="form.alphaEnd"
                type="number"
                step="0.05"
                min="0"
                max="1"
                class="w-full"
              />
              <UInput
                v-model.number="form.alphaStep"
                type="number"
                step="0.05"
                min="0.01"
                max="1"
                class="w-full"
              />
            </div>
          </div>
          <div class="space-y-2 flex flex-col">
            <span class="text-xs font-medium">Selection Mode</span>
            <USelect
              v-model="form.selectionMode"
              :items="[
                { label: 'Best EV', value: 'best_ev' },
                { label: 'Top Prob', value: 'top_prob' },
                { label: 'Top Prob Only', value: 'top_prob_only' },
                { label: 'Top Prob Always', value: 'top_prob_always' }
              ]"
            />
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

    <div v-if="matrix" class="space-y-6">
      <!-- Last Run Summary -->
      <UAlert
        v-if="sweepStats"
        variant="soft"
        color="primary"
        title="Loaded Sweep Results"
        :description="`Showing results from ${sweepStats.start_date} to ${sweepStats.end_date}. Edge: ${sweepStats.edge_range?.[0]}-${sweepStats.edge_range?.[1]}, EV: ${sweepStats.ev_range?.[0]}-${sweepStats.ev_range?.[1]}. Mode: ${sweepStats.selection_mode}. Alpha: ${selectedAlpha || (sweepStats.blend_alpha ?? 1)}. Min Bets: ${sweepStats.min_bets}`"
        icon="i-lucide-info"
      />
      <div
        v-if="alphaOptions.length"
        class="flex items-center gap-2 text-xs text-muted -mt-2"
      >
        <span>Alpha view:</span>
        <USelect
          v-model="selectedAlpha"
          :items="alphaOptions"
          @update:modelValue="selectAlpha"
        />
      </div>

      <div class="grid gap-6 lg:grid-cols-2">
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
                <tr class="max-w-12">
                  <th class="p-1 border bg-gray-50 dark:bg-gray-900">
                    EV \ Eg
                  </th>
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
                    class="p-1 border font-medium bg-gray-50 dark:bg-gray-900 text-right w-12"
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
              <div class="flex items-center gap-3">
                <span class="text-[10px] text-muted">Sorted by ROI (p05)</span>
                <UButton
                  size="xs"
                  variant="ghost"
                  :label="showAllRows ? 'Show top 8' : 'Show all'"
                  @click="showAllRows = !showAllRows"
                />
              </div>
            </div>
          </template>

          <UTable
            :data="topResults"
            :columns="columns"
            class="w-full"
            :loading="loading"
          />
        </UCard>
      </div>
    </div>

    <div
      v-else-if="!loading && results.length === 0"
      class="py-12 text-center text-muted border-2 border-dashed rounded-lg"
    >
      Adjust parameters and run sweep to find optimal betting zones.
    </div>

    <UModal
      v-model:open="detailsOpen"
      :ui="{
        overlay: 'bg-black/60 backdrop-blur-sm'
      }"
    >
      <template #content>
        <UCard v-if="selectedCell" class="shadow-xl">
          <template #header>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-xs text-muted">Sweep cell details</p>
                <h3 class="text-sm font-semibold">
                  Edge ≥ {{ selectedCell.min_edge }} • EV ≥
                  {{ selectedCell.min_ev }}
                </h3>
              </div>
              <UButton
                icon="i-lucide-x"
                variant="soft"
                color="primary"
                size="sm"
                @click="detailsOpen = false"
              />
            </div>
          </template>

          <div class="grid gap-4 text-xs">
            <div class="grid gap-2 sm:grid-cols-2">
              <div
                class="rounded-lg border border-gray-200/60 p-3 dark:border-gray-800/80"
              >
                <p class="text-[10px] uppercase text-muted">Performance</p>
                <p class="mt-1 font-semibold">
                  ROI:
                  <span
                    :class="
                      selectedCell.roi >= 0
                        ? 'text-emerald-500'
                        : 'text-red-400'
                    "
                  >
                    {{ (selectedCell.roi * 100).toFixed(2) }}%
                  </span>
                  <span class="text-muted">
                    (p05:
                    {{
                      selectedCell.roi_p05 != null
                        ? (selectedCell.roi_p05 * 100).toFixed(2)
                        : '–'
                    }}%)
                  </span>
                </p>
                <p>Bets: {{ selectedCell.bets }}</p>
                <p>
                  Profit:
                  <span
                    :class="
                      selectedCell.profit >= 0
                        ? 'text-emerald-500'
                        : 'text-red-400'
                    "
                  >
                    {{ selectedCell.profit }}
                  </span>
                </p>
              </div>
              <div
                class="rounded-lg border border-gray-200/60 p-3 dark:border-gray-800/80"
              >
                <p class="text-[10px] uppercase text-muted">Pass Rates</p>
                <p>
                  Top choice has value:
                  {{
                    formatPercent(
                      getTopPasses(selectedCell),
                      selectedCell.n_all_valid
                    )
                  }}
                </p>
                <p>
                  Any outcome has value:
                  {{
                    formatPercent(
                      getAnyPasses(selectedCell),
                      selectedCell.n_all_valid
                    )
                  }}
                </p>
                <p>Total matches: {{ selectedCell.n_all_valid }}</p>
              </div>
            </div>

            <div class="grid gap-2 sm:grid-cols-2">
              <div
                class="rounded-lg border border-gray-200/60 p-3 dark:border-gray-800/80"
              >
                <p class="text-[10px] uppercase text-muted">Placed odds</p>
                <p>Avg: {{ selectedCell.avg_odds?.toFixed(2) }}</p>
                <p>Median: {{ selectedCell.median_odds?.toFixed(2) }}</p>
                <p>P90: {{ selectedCell.p90_odds?.toFixed(2) }}</p>
                <p>
                  Avg EV:
                  <span class="text-indigo-300">{{
                    selectedCell.avg_ev?.toFixed(3)
                  }}</span>
                </p>
                <p>
                  Avg Edge:
                  <span class="text-sky-300">{{
                    selectedCell.avg_edge?.toFixed(3)
                  }}</span>
                </p>
              </div>
              <div
                class="rounded-lg border border-gray-200/60 p-3 dark:border-gray-800/80"
              >
                <p class="text-[10px] uppercase text-muted">Outcome Mixtures</p>
                <p>
                  Placed bets:
                  <span class="text-emerald-200">
                    {{ formatMix(selectedCell.stats_placed_bets?.mix) }}
                  </span>
                </p>
                <p>
                  All top choices:
                  <span class="text-sky-200">
                    {{
                      formatMix(
                        selectedCell.stats_top_prob_all_valid?.mix ??
                          selectedCell.stats_baseline_top_choice?.mix
                      )
                    }}
                  </span>
                </p>
                <p>
                  Top choices with value:
                  <span class="text-amber-200">
                    {{
                      formatMix(
                        selectedCell.stats_top_passes_gate?.mix ??
                          selectedCell.stats_value_top_choice?.mix
                      )
                    }}
                  </span>
                </p>
              </div>
            </div>

            <div
              class="rounded-lg border border-gray-200/60 p-3 dark:border-gray-800/80"
            >
              <p class="text-[10px] uppercase text-muted">Diagnostics</p>
              <div class="grid gap-2 sm:grid-cols-3">
                <div>
                  <p class="text-[10px] text-muted">Eligible matches</p>
                  <p>{{ selectedCell.n_all_valid }}</p>
                </div>
                <div>
                  <p class="text-[10px] text-muted">Top choice has value</p>
                  <p>{{ getTopPasses(selectedCell) }}</p>
                </div>
                <div>
                  <p class="text-[10px] text-muted">Any outcome has value</p>
                  <p>{{ getAnyPasses(selectedCell) }}</p>
                </div>
              </div>
              <p class="mt-2 text-[10px] text-muted">
                {{ selectedCell.all_valid_definition }}
              </p>
            </div>
          </div>
        </UCard>
      </template>
    </UModal>
  </div>
</template>
