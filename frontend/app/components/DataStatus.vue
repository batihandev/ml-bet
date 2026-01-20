<script setup lang="ts">
const config = useRuntimeConfig()
const toast = useToast()

interface FileStatus {
  exists: boolean
  size: number
  modified: number | null
}

interface DataStatus {
  raw: FileStatus
  dataset: FileStatus
  features: FileStatus
  zip: FileStatus
}

const status = ref<DataStatus | null>(null)
const loading = ref(false)

async function fetchStatus() {
  try {
    const data = await $fetch<DataStatus>(`${config.public.apiBase}/data/status`)
    status.value = data
  } catch (e) {
    console.error('Failed to fetch data status', e)
  }
}

async function unzipRaw() {
  loading.value = true
  try {
    await $fetch(`${config.public.apiBase}/data/unzip-raw`, { method: 'POST' })
  } catch (e: any) {
    toast.add({ title: 'Error', description: e.data?.detail || 'Failed to start unzip', color: 'error' })
  } finally {
    loading.value = false
  }
}

async function buildDataset() {
  loading.value = true
  try {
    await $fetch(`${config.public.apiBase}/data/build-dataset`, { method: 'POST' })
  } catch (e: any) {
    toast.add({ title: 'Error', description: e.data?.detail || 'Failed to start build', color: 'error' })
  } finally {
    loading.value = false
  }
}

async function buildFeatures() {
  loading.value = true
  try {
    await $fetch(`${config.public.apiBase}/data/build-features`, { method: 'POST' })
  } catch (e: any) {
    toast.add({ title: 'Error', description: e.data?.detail || 'Failed to start build', color: 'error' })
  } finally {
    loading.value = false
  }
}

function formatSize(bytes: number) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

function formatDate(timestamp: number | null) {
  if (!timestamp) return 'N/A'
  return new Date(timestamp * 1000).toLocaleString()
}

onMounted(() => {
  fetchStatus()
})

// Refresh status when jobs complete
const { lastEvent } = useJobEvents()
watch(lastEvent, (val) => {
  if (
    val?.type === 'dataset_build_completed' || 
    val?.type === 'features_build_completed' ||
    val?.type === 'unzip_completed'
  ) {
    fetchStatus()
  }
})
</script>

<template>
  <div class="space-y-4">
    <div v-if="status?.raw.exists === false" class="rounded-lg bg-orange-500/10 p-4 border border-orange-500/20">
      <div class="flex items-center gap-3">
        <UIcon name="i-heroicons-exclamation-triangle" class="h-5 w-5 text-orange-500" />
        <div class="text-sm text-orange-600 dark:text-orange-400">
          <span class="font-bold">Raw data missing:</span> 
          <code>data/raw/Matches.csv</code> not found. Please provide it to start the flow.
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <!-- Zip File -->
      <UCard :ui="{ body: { padding: 'p-4' } }">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider text-muted">Zip Source</span>
          <UIcon 
            :name="status?.zip.exists ? 'i-heroicons-archive-box' : 'i-heroicons-x-circle'" 
            :class="status?.zip.exists ? 'text-blue-500' : 'text-red-500'"
            class="h-5 w-5"
          />
        </div>
        <div class="text-sm font-medium mb-1 truncate">{{ status?.zip.name || 'data-2000-2025.zip' }}</div>
        <div class="text-[11px] text-muted truncate">
          {{ status?.zip.exists ? `${formatSize(status.zip.size)} • ${formatDate(status.zip.modified)}` : 'Zip missing' }}
        </div>
      </UCard>

      <!-- Raw Data -->
      <UCard :ui="{ body: { padding: 'p-4' } }">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider text-muted">Raw CSV</span>
          <UIcon 
            :name="status?.raw.exists ? 'i-heroicons-check-circle' : 'i-heroicons-x-circle'" 
            :class="status?.raw.exists ? 'text-green-500' : 'text-red-500'"
            class="h-5 w-5"
          />
        </div>
        <div class="text-sm font-medium mb-1 truncate">Matches.csv</div>
        <div class="flex items-end justify-between">
          <div class="text-[11px] text-muted truncate">
            {{ status?.raw.exists ? `${formatSize(status.raw.size)} • ${formatDate(status.raw.modified)}` : 'Needs unzip' }}
          </div>
          <UButton
            v-if="!status?.raw.exists && status?.zip.exists"
            size="2xs"
            variant="ghost"
            label="Unzip"
            :loading="loading"
            @click="unzipRaw"
          />
        </div>
      </UCard>

      <!-- Processed Dataset -->
      <UCard :ui="{ body: { padding: 'p-4' } }">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider text-muted">Dataset</span>
          <UIcon 
            :name="status?.dataset.exists ? 'i-heroicons-check-circle' : 'i-heroicons-x-circle'" 
            :class="status?.dataset.exists ? 'text-green-500' : 'text-red-500'"
            class="h-5 w-5"
          />
        </div>
        <div class="text-sm font-medium mb-1 truncate">matches.csv</div>
        <div class="flex items-end justify-between">
          <div class="text-[11px] text-muted">
            {{ status?.dataset.exists ? `${formatSize(status.dataset.size)} • ${formatDate(status.dataset.modified)}` : 'Needs build' }}
          </div>
          <UButton
            v-if="status?.raw.exists"
            size="2xs"
            variant="ghost"
            label="Build"
            :loading="loading"
            @click="buildDataset"
          />
        </div>
      </UCard>

      <!-- Engineered Features -->
      <UCard :ui="{ body: { padding: 'p-4' } }">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-semibold uppercase tracking-wider text-muted">Features</span>
          <UIcon 
            :name="status?.features.exists ? 'i-heroicons-check-circle' : 'i-heroicons-x-circle'" 
            :class="status?.features.exists ? 'text-green-500' : 'text-red-500'"
            class="h-5 w-5"
          />
        </div>
        <div class="text-sm font-medium mb-1 truncate">features.csv</div>
        <div class="flex items-end justify-between">
          <div class="text-[11px] text-muted">
            {{ status?.features.exists ? `${formatSize(status.features.size)} • ${formatDate(status.features.modified)}` : 'Needs build' }}
          </div>
          <UButton
            v-if="status?.dataset.exists"
            size="2xs"
            variant="ghost"
            label="Build"
            :loading="loading"
            @click="buildFeatures"
          />
        </div>
      </UCard>
    </div>
  </div>
</template>
