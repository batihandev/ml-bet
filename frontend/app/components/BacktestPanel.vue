<script setup lang="ts">
import { reactive, ref, h } from 'vue'
import type { TableColumn } from '@nuxt/ui'

const config = useRuntimeConfig()
const route = useRoute()

const form = reactive({
  startDate: (route.query.startDate as string) || '2024-12-30',
  endDate: (route.query.endDate as string) || '2025-06-30',
  minEdge: parseFloat(route.query.minEdge as string) || 0.05,
  minEv: parseFloat(route.query.minEv as string) || 0.0,
  stake: parseFloat(route.query.stake as string) || 1,
  kellyMult: parseFloat(route.query.kellyMult as string) || 0
})

// Default TRAINING_CUTOFF from training preset
const TRAINING_CUTOFF = ref('2024-12-29')
const DATA_END = '2025-06-30'

const minBacktestStart = computed(() => {
  const d = new Date(TRAINING_CUTOFF.value)
  d.setDate(d.getDate() + 1)
  return d.toISOString().split('T')[0]
})

const isBacktestAvailable = computed(() => {
  return TRAINING_CUTOFF.value < DATA_END
})

const backtestPresets = computed(() => {
  return [
    { label: 'Last 6 Months', start: '2024-12-30' },
    { label: 'Last 12 Months', start: '2024-06-30' },
    { label: 'Full Test', start: minBacktestStart.value }
  ]
})

function applyBacktestPreset(p: any) {
  const startStr = p.start as string | undefined
  let startSelected = startStr

  // Clamp to training_cutoff + 1 day if needed
  if (startStr && new Date(startStr) < new Date(minBacktestStart.value || 0)) {
    startSelected = minBacktestStart.value
  }

  if (startSelected) {
    form.startDate = startSelected
  }
  form.endDate = DATA_END
}

const metadataLoading = ref(false)
async function fetchModelMetadata() {
  const config = useRuntimeConfig()
  metadataLoading.value = true
  try {
    const res = await $fetch<any>(
      `${config.public.apiBase}/train/feature-importance/model_ft_1x2`
    )
    const cutoff = res.training_params?.training_cutoff_date
    if (cutoff && cutoff !== 'None') {
      TRAINING_CUTOFF.value = cutoff
      // Also sync form start if it's currently before the new cutoff
      const minStart = minBacktestStart.value
      if (
        form.startDate &&
        minStart &&
        new Date(form.startDate) < new Date(minStart)
      ) {
        form.startDate = minStart
      }
    }
  } catch (e) {
    console.error('Failed to fetch model metadata for backtest', e)
  } finally {
    metadataLoading.value = false
  }
}

onMounted(() => {
  fetchModelMetadata().then(() => {
    if (route.query.autoRun === 'true') {
      runBacktest()
    }
  })
})

const loading = ref(false)

type Summary = {
  total_bets: number
  total_staked: number
  total_profit: number
  roi: number
  avg_odds: number
  avg_ev: number
  roi_p05?: number
  roi_p95?: number
}

type MarketRow = {
  label: string
  bets: number
  wins: number
  hit_rate: number
  roi: number
}

type DivisionRow = {
  division: string
  bets: number
  wins: number
  staked: number
  profit: number
  roi: number
}

const summary = ref<Summary | null>(null)
const markets = ref<MarketRow[]>([])
const divisions = ref<DivisionRow[]>([])
const equity = ref<any[]>([])

// --- Nuxt UI v4 columns ------------------------------------------------

const marketColumns: TableColumn<MarketRow>[] = [
  { accessorKey: 'label', header: 'Market' },
  { accessorKey: 'bets', header: 'Bets' },
  { accessorKey: 'wins', header: 'Wins' },
  {
    accessorKey: 'profit',
    header: 'Profit',
    cell: ({ row }) => {
      const v = (row.getValue('profit') as number) ?? 0
      const cls = v >= 0 ? 'text-emerald-500' : 'text-red-500'
      return h('span', { class: cls }, v.toFixed(2))
    }
  },
  {
    accessorKey: 'hit_rate',
    header: 'Hit rate',
    cell: ({ row }) => {
      const v = (row.getValue('hit_rate') as number) ?? 0
      return `${(v * 100).toFixed(1)}%`
    }
  },
  {
    accessorKey: 'roi',
    header: 'ROI',
    cell: ({ row }) => {
      const v = (row.getValue('roi') as number) ?? 0
      const cls = v >= 0 ? 'text-emerald-500' : 'text-red-500'
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  }
]

const divisionColumns: TableColumn<DivisionRow>[] = [
  { accessorKey: 'division', header: 'Div' },
  { accessorKey: 'bets', header: 'Bets' },
  { accessorKey: 'wins', header: 'Wins' },
  { accessorKey: 'staked', header: 'Staked' },
  {
    accessorKey: 'profit',
    header: 'Profit',
    cell: ({ row }) => {
      const v = (row.getValue('profit') as number) ?? 0
      const cls = v >= 0 ? 'text-emerald-500' : 'text-red-500'
      return h('span', { class: cls }, v.toFixed(2))
    }
  },
  {
    accessorKey: 'roi',
    header: 'ROI',
    cell: ({ row }) => {
      const v = (row.getValue('roi') as number) ?? 0
      const cls = v >= 0 ? 'text-emerald-500' : 'text-red-500'
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  }
]

// --- Backtest call -----------------------------------------------------

async function runBacktest() {
  loading.value = true

  try {
    const res = await $fetch<{
      summary: Summary
      markets: MarketRow[]
      divisions: DivisionRow[]
      equity: any[]
    }>(`${config.public.apiBase}/backtest/ft-1x2`, {
      method: 'POST',
      body: {
        start_date: form.startDate || null,
        end_date: form.endDate || null,
        min_edge: form.minEdge,
        min_ev: form.minEv,
        stake: form.stake,
        kelly_mult: form.kellyMult
      }
    })

    summary.value = res.summary
    markets.value = res.markets ?? []
    divisions.value = res.divisions ?? []
    equity.value = res.equity ?? []
  } catch (err) {
    console.error('Backtest error', err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <UPageSection
    id="backtests"
    title="FT 1X2 backtests"
    description="Run EV-based backtests over your filtered leagues to see how the models behave with different edge thresholds and staking strategies."
  >
    <div class="space-y-6">
      <!-- Controls -->
      <UCard>
        <UForm :state="form" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-4">
            <UFormField
              label="Backtest start"
              help="First date included in the backtest."
            >
              <UInput v-model="form.startDate" type="date" />
            </UFormField>

            <UFormField
              label="Backtest end"
              help="Last date included in the backtest."
            >
              <UInput v-model="form.endDate" type="date" />
            </UFormField>

            <UFormField
              label="Min edge"
              help="p_model − p_implied threshold to place a bet."
            >
              <UInput
                v-model.number="form.minEdge"
                type="number"
                step="0.01"
                min="0"
                max="0.5"
                placeholder="0.05"
              />
            </UFormField>

            <UFormField
              label="Min EV"
              help="Expected value threshold. 0 = +EV only."
            >
              <UInput
                v-model.number="form.minEv"
                type="number"
                step="0.01"
                min="-0.5"
                max="0.5"
                placeholder="0.0"
              />
            </UFormField>

            <UFormField
              label="Stake / Kelly"
              help="Flat stake and Kelly multiplier."
            >
              <div class="flex gap-2">
                <UInput
                  v-model.number="form.stake"
                  type="number"
                  step="0.5"
                  min="0"
                  class="w-1/2"
                  placeholder="Stake"
                />
                <UInput
                  v-model.number="form.kellyMult"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  class="w-1/2"
                  placeholder="Kelly ×"
                />
              </div>
              <p class="mt-1 text-[11px] text-muted">
                Kelly = 0 → flat stake only. Kelly ≈ 0.25 is a common cap.
              </p>
            </UFormField>
          </div>

          <div class="flex items-center justify-between gap-4">
            <div class="flex items-center gap-4">
              <div class="flex gap-2">
                <UButton
                  v-for="p in backtestPresets"
                  :key="p.label"
                  size="xs"
                  variant="soft"
                  @click="applyBacktestPreset(p)"
                >
                  {{ p.label }}
                </UButton>
              </div>
              <div class="flex items-center gap-1.5 text-xs text-muted">
                <UIcon name="i-lucide-info" class="h-3.5 w-3.5" />
                <span
                  >Available from:
                  <strong :class="!isBacktestAvailable ? 'text-red-500' : ''">
                    {{
                      isBacktestAvailable ? minBacktestStart : 'No labeled data'
                    }}
                  </strong></span
                >
                <UButton
                  variant="ghost"
                  icon="i-lucide-refresh-cw"
                  size="xs"
                  :loading="metadataLoading"
                  class="-ml-1"
                  @click="fetchModelMetadata"
                />
              </div>
            </div>

            <UButton
              color="primary"
              :disabled="!isBacktestAvailable"
              :loading="loading"
              icon="i-lucide-activity"
              @click="runBacktest"
            >
              {{ isBacktestAvailable ? 'Run backtest' : 'Not available' }}
            </UButton>
          </div>
        </UForm>
      </UCard>

      <!-- Results summary -->
      <div class="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
        <UCard>
          <p class="text-xs font-medium text-muted">Total bets</p>
          <p class="mt-2 text-2xl font-semibold">
            {{ summary?.total_bets ?? '–' }}
          </p>
        </UCard>

        <UCard>
          <p class="text-xs font-medium text-muted">Avg Odds / EV</p>
          <p class="mt-2 text-2xl font-semibold">
            {{ summary ? summary.avg_odds.toFixed(2) : '–' }}
            <span class="text-xs font-normal text-muted ml-1">
              evt: {{ summary ? summary.avg_ev.toFixed(3) : '–' }}
            </span>
          </p>
        </UCard>

        <UCard>
          <p class="text-xs font-medium text-muted">Total profit</p>
          <p class="mt-2 text-2xl font-semibold">
            <span
              :class="
                (summary?.total_profit ?? 0) >= 0
                  ? 'text-emerald-500'
                  : 'text-red-500'
              "
            >
              {{ summary ? summary.total_profit.toFixed(2) : '–' }}
            </span>
          </p>
        </UCard>

        <UCard>
          <p class="text-xs font-medium text-muted">Overall ROI</p>
          <p class="mt-2 text-2xl font-semibold">
            <span
              :class="
                (summary?.roi ?? 0) >= 0 ? 'text-emerald-500' : 'text-red-500'
              "
            >
              {{ summary ? (summary.roi * 100).toFixed(1) + '%' : '–' }}
            </span>
          </p>
        </UCard>

        <UCard v-if="summary?.roi_p05 !== undefined">
          <p class="text-xs font-medium text-muted">ROI Confidence (90%)</p>
          <p class="mt-2 text-lg font-semibold">
            <span
              :class="
                summary.roi_p05 >= 0 ? 'text-emerald-500' : 'text-orange-500'
              "
            >
              {{ (summary.roi_p05 * 100).toFixed(1) }}%
            </span>
            <span class="mx-1 text-muted text-sm">—</span>
            <span v-if="summary.roi_p95 !== undefined" class="text-emerald-500">
              {{ (summary.roi_p95 * 100).toFixed(1) }}%
            </span>
          </p>
          <p class="text-[10px] text-muted">Lower bound (p05) is key.</p>
        </UCard>
      </div>

      <div class="grid gap-6 lg:grid-cols-[minmax(0,1.5fr),minmax(0,1.2fr)]">
        <!-- Markets + equity -->
        <div class="space-y-4">
          <UCard>
            <h3 class="mb-3 text-sm font-semibold">Per-market performance</h3>

            <PerformanceTable
              :data="markets"
              :columns="marketColumns"
              empty-message="Run a backtest to see per-market stats."
            />
          </UCard>

          <UCard>
            <h3 class="mb-3 text-sm font-semibold">Equity curve</h3>
            <EquityCurve :equity="equity" />
          </UCard>
        </div>

        <!-- Per-division breakdown -->
        <UCard>
          <h3 class="mb-3 text-sm font-semibold">Per-division breakdown</h3>

          <PerformanceTable
            :data="divisions"
            :columns="divisionColumns"
            empty-message="Once wired to the backend, this table will show which leagues actually make money."
          />
        </UCard>
      </div>
    </div>
  </UPageSection>
</template>
