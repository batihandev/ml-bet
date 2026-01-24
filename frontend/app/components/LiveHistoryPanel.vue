<script setup lang="ts">
import { reactive, ref, h } from 'vue'
import type { TableColumn } from '@nuxt/ui'

const config = useRuntimeConfig()

const form = reactive({
  division: 'E0',
  matchDate: '2025-12-20',
  homeTeam: '',
  awayTeam: '',
  oddHome: 1.8,
  oddDraw: 3.6,
  oddAway: 4.2
})

const loading = ref(false)
const metaLoading = ref(false)
const teams = ref<string[]>([])
const divisions = ref<string[]>([])

async function fetchMeta() {
  metaLoading.value = true
  try {
    const [teamRes, divRes] = await Promise.all([
      $fetch<{ teams: string[] }>(`${config.public.apiBase}/meta/teams`),
      $fetch<{ divisions: string[] }>(`${config.public.apiBase}/meta/divisions`)
    ])
    teams.value = teamRes.teams
    divisions.value = divRes.divisions
  } catch (err) {
    console.error('Failed to fetch metadata', err)
  } finally {
    metaLoading.value = false
  }
}

onMounted(fetchMeta)

type MatchInfo = {
  division: string
  match_date: string
  home_team: string
  away_team: string
  odd_home: number | null
  odd_draw: number | null
  odd_away: number | null
}

type PredictionMetric = {
  outcome: string
  prob: number
  pimp: number
  edge: number
  ev: number
  odds: number
}

type PredictionResult = {
  match: MatchInfo
  probabilities: Record<string, number>
  implied_probabilities: Record<string, number>
  recommendation: string
  best_ev: number | null
  metrics: PredictionMetric[]
}

const matchInfo = ref<MatchInfo | null>(null)
const recommendation = ref<string>('no bet')
const bestEv = ref<number | null>(null)
const metrics = ref<PredictionMetric[]>([])

const metricColumns: TableColumn<PredictionMetric>[] = [
  { accessorKey: 'outcome', header: 'Outcome' },
  {
    accessorKey: 'odds',
    header: 'Odds',
    cell: ({ row }) => (row.getValue('odds') as number).toFixed(2)
  },
  {
    accessorKey: 'prob',
    header: 'Model Prob',
    cell: ({ row }) => {
      const v = row.getValue('prob') as number
      return `${(v * 100).toFixed(1)}%`
    }
  },
  {
    accessorKey: 'pimp',
    header: 'Fair Prob',
    cell: ({ row }) => {
      const v = row.getValue('pimp') as number
      return v ? `${(v * 100).toFixed(1)}%` : '-'
    }
  },
  {
    accessorKey: 'edge',
    header: 'Edge',
    cell: ({ row }) => {
      const v = row.getValue('edge') as number
      const cls =
        v > 0.05
          ? 'text-emerald-500 font-semibold'
          : v < 0
            ? 'text-rose-500'
            : ''
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  },
  {
    accessorKey: 'ev',
    header: 'EV',
    cell: ({ row }) => {
      const v = row.getValue('ev') as number
      const cls =
        v > 0.1
          ? 'text-emerald-500 font-semibold'
          : v < 0
            ? 'text-rose-500'
            : ''
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}`)
    }
  }
]

async function runLivePrediction() {
  const toast = useToast()
  loading.value = true
  matchInfo.value = null
  recommendation.value = 'no bet'
  bestEv.value = null
  metrics.value = []

  try {
    const res = await $fetch<PredictionResult>(
      `${config.public.apiBase}/predict/live-with-history`,
      {
        method: 'POST',
        body: {
          division: form.division.trim(),
          match_date: form.matchDate,
          home_team: form.homeTeam.trim(),
          away_team: form.awayTeam.trim(),
          odd_home: form.oddHome,
          odd_draw: form.oddDraw,
          odd_away: form.oddAway
        }
      }
    )

    matchInfo.value = res.match
    recommendation.value = res.recommendation
    bestEv.value = res.best_ev
    metrics.value = res.metrics ?? []

    toast.add({
      title: 'Prediction ready',
      description: `Recommendation: ${res.recommendation.toUpperCase()}`,
      color: res.recommendation !== 'no bet' ? 'success' : 'primary'
    })
  } catch (err: any) {
    console.error('Live-with-history predict error', err)
    toast.add({
      title: 'Prediction failed',
      description: err?.data?.detail || 'Check teams / division / date.',
      color: 'error'
    })
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <UPageSection
    id="live-history"
    title="Predict live match using historical form & H2H"
    description="Enter a future match with current odds. Backend will recompute form and head-to-head using all past data and return fresh model probabilities."
  >
    <div class="space-y-6">
      <UCard>
        <UForm :state="form" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-4">
            <UFormField label="Division" help="League code (e.g. E0, D1, T1).">
              <USelectMenu
                v-model="form.division"
                :items="divisions"
                :loading="metaLoading"
                searchable
              />
            </UFormField>

            <UFormField label="Match date" help="YYYY-MM-DD">
              <UInput v-model="form.matchDate" type="date" />
            </UFormField>

            <UFormField label="Home team">
              <USelectMenu
                v-model="form.homeTeam"
                :items="teams"
                :loading="metaLoading"
                searchable
                placeholder="Search home team..."
              />
            </UFormField>

            <UFormField label="Away team">
              <USelectMenu
                v-model="form.awayTeam"
                :items="teams"
                :loading="metaLoading"
                searchable
                placeholder="Search away team..."
              />
            </UFormField>
          </div>

          <div class="grid gap-4 md:grid-cols-3">
            <UFormField label="1 odds (home)">
              <UInput v-model.number="form.oddHome" type="number" step="0.01" />
            </UFormField>
            <UFormField label="X odds (draw)">
              <UInput v-model.number="form.oddDraw" type="number" step="0.01" />
            </UFormField>
            <UFormField label="2 odds (away)">
              <UInput v-model.number="form.oddAway" type="number" step="0.01" />
            </UFormField>
          </div>

          <div class="flex items-center justify-between gap-4">
            <p class="text-xs text-muted">
              Only historical matches before this date are used to build form
              and H2H. Result columns for this match are left empty.
            </p>

            <UButton
              color="primary"
              :loading="loading"
              icon="i-lucide-brain-circuit"
              @click="runLivePrediction"
            >
              Predict live
            </UButton>
          </div>
        </UForm>
      </UCard>

      <div v-if="matchInfo" class="space-y-4">
        <UCard>
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p class="text-xs font-medium text-muted">Match</p>
              <p class="mt-1 text-lg font-semibold">
                {{ matchInfo.home_team }} vs {{ matchInfo.away_team }}
              </p>
              <p class="text-xs text-muted">
                Div {{ matchInfo.division }} • {{ matchInfo.match_date }}
              </p>
            </div>

            <div
              v-if="
                matchInfo.odd_home && matchInfo.odd_draw && matchInfo.odd_away
              "
              class="flex gap-4 text-xs"
            >
              <div>
                <p class="text-muted">1</p>
                <p class="font-medium">
                  {{ matchInfo.odd_home.toFixed(2) }}
                </p>
              </div>
              <div>
                <p class="text-muted">X</p>
                <p class="font-medium">
                  {{ matchInfo.odd_draw.toFixed(2) }}
                </p>
              </div>
              <div>
                <p class="text-muted">2</p>
                <p class="font-medium">
                  {{ matchInfo.odd_away.toFixed(2) }}
                </p>
              </div>
            </div>
          </div>
        </UCard>

        <div class="grid gap-4 md:grid-cols-1">
          <UCard
            v-if="recommendation !== 'no bet'"
            class="border-2 border-emerald-500/50 bg-emerald-500/5"
          >
            <div class="flex items-center justify-between">
              <div>
                <p
                  class="text-xs font-medium text-emerald-600 dark:text-emerald-400"
                >
                  Production Recommendation
                </p>
                <p
                  class="mt-1 text-2xl font-bold text-emerald-700 dark:text-emerald-300"
                >
                  BET ON: {{ recommendation.toUpperCase() }}
                </p>
              </div>
              <div class="text-right">
                <p class="text-xs font-medium text-muted">Estimated EV</p>
                <p class="text-xl font-bold">{{ bestEv?.toFixed(2) }}</p>
              </div>
            </div>
          </UCard>

          <UCard v-else class="border-2 border-slate-500/30 bg-slate-500/5">
            <div class="text-center py-2">
              <p class="text-sm font-medium text-muted">No Recommendation</p>
              <p class="text-xs text-muted/70">
                Metrics do not meet production thresholds
              </p>
            </div>
          </UCard>

          <UCard>
            <h3 class="mb-3 text-sm font-semibold">1X2 Model Metrics</h3>
            <UTable :data="metrics" :columns="metricColumns" />
          </UCard>
        </div>
      </div>

      <p v-else class="text-xs text-muted">
        Enter a future match that uses teams already present in your historic
        data (so form and H2H can be computed). Then click
        <span class="font-medium">Predict live</span>.
      </p>
    </div>
  </UPageSection>
</template>
