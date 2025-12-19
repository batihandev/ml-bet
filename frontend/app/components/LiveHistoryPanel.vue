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

type MatchInfo = {
  division: string
  match_date: string
  home_team: string
  away_team: string
  odd_home: number | null
  odd_draw: number | null
  odd_away: number | null
}

type ProbRow = {
  key: string
  label: string
  prob: number
}

const matchInfo = ref<MatchInfo | null>(null)
const htftRows = ref<ProbRow[]>([])
const ftRows = ref<ProbRow[]>([])

const probColumns: TableColumn<ProbRow>[] = [
  { accessorKey: 'label', header: 'Outcome' },
  {
    accessorKey: 'prob',
    header: 'Model prob',
    cell: ({ row }) => {
      const v = row.getValue('prob') as number
      const cls =
        v >= 0.45
          ? 'font-semibold text-emerald-500'
          : v <= 0.2
          ? 'text-muted'
          : ''
      return h('span', { class: cls }, `${(v * 100).toFixed(1)}%`)
    }
  }
]

async function runLivePrediction() {
  const toast = useToast()
  loading.value = true
  matchInfo.value = null
  htftRows.value = []
  ftRows.value = []

  try {
    const res = await $fetch<{
      match: MatchInfo
      htft: ProbRow[]
      ft_1x2: ProbRow[]
    }>(`${config.public.apiBase}/predict/live-with-history`, {
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
    })

    matchInfo.value = res.match
    htftRows.value = res.htft ?? []
    ftRows.value = res.ft_1x2 ?? []

    toast.add({
      title: 'Live prediction ready',
      description:
        'Built features using historical form & H2H and ran all models.',
      color: 'success'
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
              <UInput v-model="form.division" />
            </UFormField>

            <UFormField label="Match date" help="YYYY-MM-DD">
              <UInput v-model="form.matchDate" type="date" />
            </UFormField>

            <UFormField label="Home team">
              <UInput v-model="form.homeTeam" placeholder="Liverpool" />
            </UFormField>

            <UFormField label="Away team">
              <UInput v-model="form.awayTeam" placeholder="Beşiktaş" />
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

        <div class="grid gap-4 md:grid-cols-2">
          <UCard>
            <h3 class="mb-3 text-sm font-semibold">HT/FT patterns</h3>
            <UTable :data="htftRows" :columns="probColumns" />
          </UCard>

          <UCard>
            <h3 class="mb-3 text-sm font-semibold">FT 1X2</h3>
            <UTable :data="ftRows" :columns="probColumns" />
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
