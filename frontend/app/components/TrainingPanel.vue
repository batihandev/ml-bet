<script setup lang="ts">
import { reactive, ref, computed } from 'vue'

const form = reactive({
  trainStart: '2020-06-30',
  trainEnd: '2025-06-30',
  cutoffDate: '2024-12-30',
  nEstimators: 300,
  maxDepth: 8,
  minSamplesLeaf: 50
})

const trainingPresets = [
  {
    label: 'Last 5 Years',
    start: '2020-06-30',
    end: '2025-06-30',
    val: '2024-12-30'
  },
  {
    label: 'Last 10 Years',
    start: '2015-06-30',
    end: '2025-06-30',
    val: '2024-06-30'
  },
  {
    label: 'Start to End',
    start: '2000-01-01',
    end: '2025-06-30',
    val: '2023-06-30'
  }
]

function applyTrainingPreset(p: any) {
  form.trainStart = p.start
  form.trainEnd = p.end
  form.cutoffDate = p.val
}

const loading = ref(false)
const statusMessage = ref<string | null>(null)

const modelMeta = ref<any>(null)
const importanceLoading = ref(false)
const availableModels = ref<string[]>([])
const selectedModel = ref('model_ft_1x2')

// Compute the day before cutoff for display
const lastTrainDate = computed(() => {
  if (!form.cutoffDate) return 'auto'
  const d = new Date(form.cutoffDate)
  d.setDate(d.getDate() - 1)
  return d.toISOString().split('T')[0]
})

async function runTraining() {
  const config = useRuntimeConfig()
  loading.value = true
  statusMessage.value = null

  try {
    await $fetch(`${config.public.apiBase}/train`, {
      method: 'POST',
      body: {
        train_start: form.trainStart || null,
        train_end: form.trainEnd || null,
        cutoff_date: form.cutoffDate || null,
        n_estimators: form.nEstimators,
        max_depth: form.maxDepth,
        min_samples_leaf: form.minSamplesLeaf
      }
    })

    statusMessage.value = 'Training job dispatched.'
    setTimeout(() => {
      fetchImportance(selectedModel.value)
      fetchAvailableModels()
    }, 2000)
  } catch (e: any) {
    statusMessage.value = e?.data?.detail || 'Training failed to dispatch.'
  } finally {
    loading.value = false
  }
}

async function fetchAvailableModels() {
  const config = useRuntimeConfig()
  try {
    const res = await $fetch<{ models: string[] }>(
      `${config.public.apiBase}/train/models`
    )
    availableModels.value = res.models
    if (!selectedModel.value && res.models.length) {
      selectedModel.value = res.models[0] || ''
    }
  } catch (e) {
    console.error('Failed to fetch models:', e)
  }
}

async function fetchImportance(modelName: string) {
  if (!modelName) return
  const config = useRuntimeConfig()
  importanceLoading.value = true
  try {
    const res = await $fetch<any>(
      `${config.public.apiBase}/train/feature-importance/${modelName}`
    )
    modelMeta.value = res
  } catch (e) {
    console.error('Failed to fetch meta:', e)
    modelMeta.value = null
  } finally {
    importanceLoading.value = false
  }
}

watch(selectedModel, (newModel) => {
  if (newModel) fetchImportance(newModel)
})

onMounted(() => {
  fetchAvailableModels().then(() => {
    fetchImportance(selectedModel.value)
  })
})
</script>

<template>
  <UPageSection
    id="models"
    title="Model training window"
    description="Configure your historical window and cutoff date for training the Random Forest models."
  >
    <div class="grid gap-6 lg:grid-cols-[minmax(0,2fr),minmax(0,1.4fr)]">
      <!-- Left: form -->
      <UCard class="space-y-4">
        <UForm :state="form" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-3">
            <UFormField
              label="Data start date"
              help="First match loaded into memory."
            >
              <UInput
                v-model="form.trainStart"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>

            <UFormField
              label="Data end date"
              help="Last match loaded into memory."
            >
              <UInput
                v-model="form.trainEnd"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>

            <UFormField
              label="Validation starts at"
              help="First match used for validation (not training)."
            >
              <UInput
                v-model="form.cutoffDate"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>
          </div>

          <!-- Hyperparameters -->
          <div
            class="grid gap-4 md:grid-cols-3 pt-4 border-t border-gray-200 dark:border-gray-800"
          >
            <UFormField
              label="Estimators"
              help="Number of trees (n_estimators)."
            >
              <UInput v-model="form.nEstimators" type="number" step="50" />
            </UFormField>

            <UFormField label="Max Depth" help="Max tree depth (max_depth).">
              <UInput v-model="form.maxDepth" type="number" step="1" />
            </UFormField>

            <UFormField
              label="Min Samples"
              help="Minimum samples per leaf (min_samples_leaf)."
            >
              <UInput v-model="form.minSamplesLeaf" type="number" step="5" />
            </UFormField>
          </div>

          <!-- Live summary -->
          <div
            class="rounded-lg bg-gray-100 dark:bg-gray-800 p-3 text-xs space-y-1"
          >
            <div class="flex justify-between">
              <span class="text-muted">Training:</span>
              <span class="font-mono"
                >{{ form.trainStart || 'auto' }} → {{ lastTrainDate }}</span
              >
            </div>
            <div class="flex justify-between">
              <span class="text-muted">Validation:</span>
              <span class="font-mono"
                >{{ form.cutoffDate || 'auto' }} →
                {{ form.trainEnd || 'auto' }}</span
              >
            </div>
          </div>

          <div class="flex flex-wrap items-center justify-between gap-4">
            <div class="flex gap-2">
              <UButton
                v-for="p in trainingPresets"
                :key="p.label"
                size="xs"
                variant="soft"
                @click="applyTrainingPreset(p)"
              >
                {{ p.label }}
              </UButton>
            </div>

            <UButton
              color="primary"
              :loading="loading"
              icon="i-lucide-play-circle"
              @click="runTraining"
            >
              Run training
            </UButton>
          </div>

          <p v-if="statusMessage" class="text-xs text-muted">
            {{ statusMessage }}
          </p>
        </UForm>
      </UCard>

      <!-- Right: explainer/metrics -->
      <div class="space-y-6">
        <UCard v-if="modelMeta?.metrics" class="space-y-3">
          <template #header>
            <div class="flex items-center justify-between">
              <h3 class="text-md font-semibold">Trained Data Metrics</h3>
              <div class="flex items-center gap-2">
                <UBadge color="success" variant="subtle" size="md">
                  Acc: {{ (modelMeta.metrics.accuracy * 100).toFixed(1) }}%
                </UBadge>
                <UButton
                  variant="ghost"
                  icon="i-lucide-refresh-cw"
                  size="xs"
                  :loading="importanceLoading"
                  @click="fetchImportance(selectedModel)"
                />
              </div>
            </div>
          </template>

          <div class="text-sm space-y-4">
            <div class="grid grid-cols-2 gap-2">
              <div class="p-2 bg-gray-50 dark:bg-gray-800/50 rounded">
                <p class="text-muted">Train samples</p>
                <p class="font-bold">{{ modelMeta.metrics.n_train }}</p>
              </div>
              <div class="p-2 bg-gray-50 dark:bg-gray-800/50 rounded">
                <p class="text-muted">Val samples</p>
                <p class="font-bold">{{ modelMeta.metrics.n_val }}</p>
              </div>
            </div>

            <table class="w-full text-sm border-collapse">
              <thead>
                <tr class="border-b border-gray-200 dark:border-gray-800">
                  <th class="text-left py-1">Class</th>
                  <th class="text-right">Prec</th>
                  <th class="text-right">Rec</th>
                  <th class="text-right">F1</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(v, k) in modelMeta.metrics.classification_report"
                  :key="String(k)"
                  class="border-b border-gray-100 dark:border-gray-900 last:border-0"
                >
                  <template v-if="['0', '1', '2'].includes(String(k))">
                    <td class="py-1 font-medium">
                      {{
                        String(k) === '0'
                          ? 'Home'
                          : String(k) === '1'
                            ? 'Draw'
                            : 'Away'
                      }}
                    </td>
                    <td class="text-right">{{ v.precision.toFixed(2) }}</td>
                    <td class="text-right">{{ v.recall.toFixed(2) }}</td>
                    <td class="text-right">
                      {{ v.f1_score?.toFixed(2) || v['f1-score']?.toFixed(2) }}
                    </td>
                  </template>
                </tr>
              </tbody>
            </table>
          </div>
        </UCard>

        <UCard class="space-y-3">
          <h3 class="text-sm font-semibold">How splitting works</h3>
          <p class="text-xs text-muted">
            The code in
            <code class="font-mono text-[11px]">production/train.py</code> does:
          </p>
          <pre
            class="text-[11px] bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto font-mono"
          >
train_mask = df["match_date"] &lt; cutoff
val_mask   = df["match_date"] >= cutoff</pre
          >
          <ul class="space-y-2 text-xs text-muted">
            <li>
              <span class="font-medium text-emerald-600 dark:text-emerald-400"
                >Training:</span
              >
              All matches <strong>before</strong> the cutoff date.
            </li>
            <li>
              <span class="font-medium text-amber-600 dark:text-amber-400"
                >Validation:</span
              >
              All matches <strong>on or after</strong> the cutoff date.
            </li>
          </ul>
        </UCard>
      </div>
    </div>

    <!-- Feature Importance Section -->
    <UCard class="mt-8">
      <template #header>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2 mr-4">
              <UIcon
                name="i-lucide-bar-chart-3"
                class="h-5 w-5 text-emerald-500"
              />
              <h3 class="text-sm font-semibold">Feature Importance</h3>
            </div>

            <USelectMenu
              v-model="selectedModel"
              :items="availableModels"
              placeholder="Select model"
              class="w-64"
            />
          </div>
          <UButton
            variant="ghost"
            icon="i-lucide-refresh-cw"
            size="xs"
            :loading="importanceLoading"
            @click="fetchImportance(selectedModel)"
          >
            Refresh
          </UButton>
        </div>
      </template>
      <FeatureImportance
        :data="modelMeta?.feature_importance || []"
        :loading="importanceLoading"
      />
    </UCard>
  </UPageSection>
</template>
