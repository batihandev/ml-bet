<script setup lang="ts" generic="T">
import type { TableColumn } from '@nuxt/ui'

defineProps<{
  data: T[]
  columns: TableColumn<any>[]
  emptyMessage?: string
}>()
</script>

<template>
  <div>
    <UTable
      v-if="data.length"
      :data="data"
      :columns="columns"
    />
    <p v-else class="text-xs text-muted">
      {{ emptyMessage || 'No data available.' }}
    </p>
  </div>
</template>
