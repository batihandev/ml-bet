<script setup lang="ts">
import { reactive, ref, h } from 'vue'
import type { TableColumn } from '@nuxt/ui'

const config = useRuntimeConfig()

const form = reactive({
  startDate: '2024-12-31',
  endDate: '2025-06-01',
  minEdge: 0.015,
  stake: 1,
  kellyMult: 0
})

const loading = ref(false)

type Summary = {
  total_bets: number
  total_staked: number
  total_profit: number
  overall_roi: number
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
  { accessorKey: 'profit', header: 'Profit' },
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
            <p class="text-xs text-muted">
              Dates here control only the backtest window. Training window is
              configured above.
            </p>

            <UButton
              color="primary"
              :loading="loading"
              icon="i-lucide-activity"
              @click="runBacktest"
            >
              Run backtest
            </UButton>
          </div>
        </UForm>
      </UCard>

      <!-- Results summary -->
      <div class="grid gap-4 md:grid-cols-3">
        <UCard>
          <p class="text-xs font-medium text-muted">Total bets</p>
          <p class="mt-2 text-2xl font-semibold">
            {{ summary?.total_bets ?? '–' }}
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
                (summary?.overall_roi ?? 0) >= 0
                  ? 'text-emerald-500'
                  : 'text-red-500'
              "
            >
              {{ summary ? (summary.overall_roi * 100).toFixed(1) + '%' : '–' }}
            </span>
          </p>
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
