<script setup lang="ts">
import { reactive, ref } from 'vue'

const form = reactive({
  trainStart: '',
  trainEnd: '',
  cutoffDate: ''
})

const loading = ref(false)
const statusMessage = ref<string | null>(null)

async function runTraining() {
  loading.value = true
  statusMessage.value = null

  try {
    // TODO: hook to backend later, e.g.
    // const config = useRuntimeConfig()
    // await $fetch(`${config.public.apiBase}/train`, {
    //   method: 'POST',
    //   body: {
    //     TRAIN_START_DATE: form.trainStart || null,
    //     TRAIN_END_DATE: form.trainEnd || null,
    //     FIXED_CUTOFF_DATE: form.cutoffDate || null
    //   }
    // })

    // for now just simulate success
    await new Promise((resolve) => setTimeout(resolve, 600))
    statusMessage.value =
      'Training job dispatched (mock). Wire backend when ready.'
  } catch (e) {
    statusMessage.value = 'Training failed (see console / backend logs).'
    // console.error(e)
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
              label="Train start date"
              help="First match date included in training (TRAIN_START_DATE)."
            >
              <UInput
                v-model="form.trainStart"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>

            <UFormField
              label="Train end date"
              help="Last match date included in training (TRAIN_END_DATE)."
            >
              <UInput
                v-model="form.trainEnd"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>

            <UFormField
              label="Validation cutoff"
              help="Split date for time-based validation (FIXED_CUTOFF_DATE)."
            >
              <UInput
                v-model="form.cutoffDate"
                type="date"
                placeholder="YYYY-MM-DD"
              />
            </UFormField>
          </div>

          <div class="flex items-center justify-between gap-4">
            <p class="text-xs text-muted">
              Leave fields empty to let the backend pick defaults (full range /
              quantile cutoff).
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
        <h3 class="text-sm font-semibold">How these dates are used</h3>
        <ul class="space-y-2 text-xs text-muted">
          <li>
            <span class="font-medium">Train start:</span>
            lower bound for
            <code class="font-mono text-[11px]">match_date</code> in your
            feature set.
          </li>
          <li>
            <span class="font-medium">Train end:</span>
            upper bound for
            <code class="font-mono text-[11px]">match_date</code> in your
            feature set.
          </li>
          <li>
            <span class="font-medium">Validation cutoff:</span>
            everything before this date is training, everything on/after is used
            as validation in
            <code class="font-mono text-[11px]">train_model.py</code>.
          </li>
          <li>
            <span class="font-medium">Divisions / leagues:</span>
            filtering to
            <code class="font-mono text-[11px]">ALLOWED_DIVISIONS</code>
            happens inside the Python code; this UI only controls dates.
          </li>
        </ul>
      </UCard>
    </div>
  </UPageSection>
</template>
