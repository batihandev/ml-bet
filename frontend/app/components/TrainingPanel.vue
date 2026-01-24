<script setup lang="ts">
import { reactive, ref, computed } from 'vue'

const form = reactive({
  trainStart: '2020-06-30',
  trainingCutoffDate: '2024-12-29',
  oofCalibration: true,
  calibrationMethod: 'none',
  oofStep: '1 month',
  oofMinTrainSpan: '24 months',
  backtestStart: '2024-12-30',
  backtestEnd: '2025-06-30',
  nEstimators: 300,
  maxDepth: 8,
  minSamplesLeaf: 50
})

const trainingPresets = [
  {
    label: 'Eval 5y + 6m test',
    trainStart: '2020-06-30',
    trainingCutoffDate: '2024-12-29',
    backtestStart: '2024-12-30',
    oofCalibration: true,
    calibrationMethod: 'isotonic'
  },
  {
    label: 'Eval 5y + 12m test',
    trainStart: '2020-06-30',
    trainingCutoffDate: '2024-06-29',
    backtestStart: '2024-06-30',
    oofCalibration: true,
    calibrationMethod: 'isotonic'
  },
  {
    label: 'Eval 10y + 6m test',
    trainStart: '2015-06-30',
    trainingCutoffDate: '2024-12-29',
    backtestStart: '2024-12-30',
    oofCalibration: true,
    calibrationMethod: 'isotonic'
  },
  {
    label: 'Deploy (All)',
    trainStart: '2020-06-30',
    trainingCutoffDate: '2025-06-30',
    backtestStart: '',
    oofCalibration: true,
    calibrationMethod: 'none'
  }
]

function applyTrainingPreset(p: any) {
  form.trainStart = p.trainStart
  form.trainingCutoffDate = p.trainingCutoffDate
  form.oofCalibration = p.oofCalibration
  if (p.calibrationMethod) form.calibrationMethod = p.calibrationMethod
  if (p.backtestStart) {
    form.backtestStart = p.backtestStart
  } else {
    form.backtestStart = ''
  }
}

const loading = ref(false)
const statusMessage = ref<string | null>(null)

const modelMeta = ref<any>(null)
const importanceLoading = ref(false)
const availableModels = ref<string[]>([])
const selectedModel = ref('model_ft_1x2')

const backtestEligibleFrom = computed(() => {
  if (!form.trainingCutoffDate) return ''
  const d = new Date(form.trainingCutoffDate)
  d.setDate(d.getDate() + 1)
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
        training_cutoff_date: form.trainingCutoffDate || null,
        oof_calibration: form.oofCalibration,
        calibration_method: form.oofCalibration
          ? form.calibrationMethod
          : 'none',
        oof_step: form.oofStep,
        oof_min_train_span: form.oofMinTrainSpan,
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
    title="Model training & OOF Calibration"
    description="Configure your historical window and forward-chaining OOF calibration."
  >
    <div class="grid gap-6 lg:grid-cols-[minmax(0,2fr),minmax(0,1.4fr)]">
      <!-- Left: form -->
      <UCard class="space-y-4">
        <UForm :state="form" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-2">
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
              label="Training cutoff date"
              help="Last labeled match date included in training and OOF."
            >
              <UInput
                v-model="form.trainingCutoffDate"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>
          </div>

          <div
            class="grid gap-4 md:grid-cols-3 pt-4 border-t border-gray-200 dark:border-gray-800"
          >
            <UFormField
              label="OOF Calibration"
              help="Enable forward-chaining calibration."
            >
              <USwitch v-model="form.oofCalibration" />
            </UFormField>

            <UFormField
              label="Calibration Method"
              help="Method to calibrate probabilities."
              :disabled="!form.oofCalibration"
            >
              <USelect
                v-model="form.calibrationMethod"
                :items="[
                  { label: 'None (Identity)', value: 'none' },
                  { label: 'Isotonic (Clip+Norm)', value: 'isotonic' }
                ]"
              />
            </UFormField>

            <UFormField
              label="OOF Fold Step"
              help="Size of each OOF test block."
              :disabled="!form.oofCalibration"
            >
              <UInput v-model="form.oofStep" placeholder="1 month" />
            </UFormField>

            <UFormField
              label="OOF Min Train"
              help="Minimum history before first OOF fold."
              :disabled="!form.oofCalibration"
            >
              <UInput v-model="form.oofMinTrainSpan" placeholder="24 months" />
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
              <span class="text-muted">Training Range:</span>
              <span class="font-mono"
                >{{ form.trainStart }} → {{ form.trainingCutoffDate }}</span
              >
            </div>
            <div class="flex justify-between">
              <span class="text-muted">OOF Mode:</span>
              <span
                class="font-mono text-emerald-500"
                v-if="form.oofCalibration"
                >Enabled ({{ form.oofStep }} steps)</span
              >
              <span class="font-mono text-gray-500" v-else>Disabled</span>
            </div>
            <div
              class="flex justify-between border-t border-gray-200 dark:border-gray-700 mt-1 pt-1"
            >
              <span class="text-muted">Backtest Eligible From:</span>
              <span class="font-mono text-blue-500 font-bold">{{
                backtestEligibleFrom
              }}</span>
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
              <h3 class="text-md font-semibold">OOF Calibration Metrics</h3>
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
                <p class="text-muted">OOF Rows</p>
                <p class="font-bold">{{ modelMeta.metrics.n_oof_rows }}</p>
              </div>
              <div class="p-2 bg-gray-50 dark:bg-gray-800/50 rounded">
                <p class="text-muted">OOF Folds</p>
                <p class="font-bold">{{ modelMeta.metrics.n_folds }}</p>
              </div>
              <div
                class="p-2 bg-gray-50 dark:bg-gray-800/50 rounded col-span-2"
              >
                <p class="text-muted">Brier Score / Log Loss</p>
                <p class="font-bold">
                  {{ modelMeta.metrics.brier_score?.toFixed(4) }} /
                  {{ modelMeta.metrics.log_loss?.toFixed(4) }}
                </p>
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
          <h3 class="text-sm font-semibold">How OOF works</h3>
          <p class="text-xs text-muted">
            Instead of a fixed window, we use forward-chaining:
          </p>
          <ul class="space-y-2 text-xs text-muted">
            <li>
              <span class="font-medium text-emerald-600 dark:text-emerald-400"
                >CHRONO FOLDS:</span
              >
              Models are trained on past data to predict a future "fold" block.
            </li>
            <li>
              <span class="font-medium text-amber-600 dark:text-amber-400"
                >CALIBRATION:</span
              >
              A calibrator is fit on aggregated OOF predictions.
            </li>
            <li>
              <span class="font-medium text-blue-600 dark:text-blue-400"
                >FINAL MODEL:</span
              >
              Trained on ALL data up to {{ form.trainingCutoffDate }}.
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
