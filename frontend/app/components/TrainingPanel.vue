<script setup lang="ts">
import { reactive, ref, computed } from 'vue'

const form = reactive({
  trainStart: '2020-01-01',
  trainEnd: '2025-06-30',
  cutoffDate: '2024-07-01'
})

const loading = ref(false)
const statusMessage = ref<string | null>(null)

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
        cutoff_date: form.cutoffDate || null
      }
    })

    statusMessage.value = 'Training job dispatched. Watch for status in the banner/toast.'
  } catch (e: any) {
    statusMessage.value = e?.data?.detail || 'Training failed to dispatch.'
  } finally {
    loading.value = false
  }
}
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

          <!-- Live summary -->
          <div class="rounded-lg bg-gray-100 dark:bg-gray-800 p-3 text-xs space-y-1">
            <div class="flex justify-between">
              <span class="text-muted">Training:</span>
              <span class="font-mono">{{ form.trainStart || 'auto' }} → {{ lastTrainDate }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted">Validation:</span>
              <span class="font-mono">{{ form.cutoffDate || 'auto' }} → {{ form.trainEnd || 'auto' }}</span>
            </div>
          </div>

          <div class="flex items-center justify-between gap-4">
            <p class="text-xs text-muted">
              Leave fields empty to let the backend pick defaults.
            </p>

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

      <!-- Right: explainer -->
      <UCard class="space-y-3">
        <h3 class="text-sm font-semibold">How splitting works</h3>
        <p class="text-xs text-muted">
          The code in <code class="font-mono text-[11px]">train_model.py</code> does:
        </p>
        <pre class="text-[11px] bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto font-mono">train_mask = df["match_date"] &lt; cutoff
val_mask   = df["match_date"] >= cutoff</pre>
        <ul class="space-y-2 text-xs text-muted">
          <li>
            <span class="font-medium text-emerald-600 dark:text-emerald-400">Training:</span>
            All matches <strong>before</strong> the cutoff date.
          </li>
          <li>
            <span class="font-medium text-amber-600 dark:text-amber-400">Validation:</span>
            All matches <strong>on or after</strong> the cutoff date.
          </li>
          <li class="pt-1 border-t border-gray-200 dark:border-gray-700">
            <span class="font-medium">No overlap:</span>
            Validation data is never seen during training.
          </li>
        </ul>
      </UCard>
    </div>
  </UPageSection>
</template>
