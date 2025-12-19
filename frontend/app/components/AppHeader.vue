<script setup lang="ts">
const links = [
  { label: 'Dashboard', to: '/', icon: 'i-lucide-home' },
  { label: 'Models', to: '#models', icon: 'i-lucide-brain' },
  { label: 'Backtests', to: '#backtests', icon: 'i-lucide-line-chart' }
]
</script>

<template>
  <UHeader class="border-b border-gray-200/60 dark:border-gray-800/80">
    <template #left>
      <NuxtLink to="/" class="flex items-center gap-2">
        <!-- Use your existing AppLogo if you like -->
        <div
          class="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500 text-xs font-semibold text-white"
        >
          FE
        </div>
        <span class="text-sm font-semibold tracking-tight">
          Football Edge Lab
        </span>
      </NuxtLink>

      <nav class="ml-6 hidden gap-1 md:flex">
        <UButton
          v-for="link in links"
          :key="link.to"
          :to="link.to"
          color="neutral"
          variant="ghost"
          size="xs"
          :icon="link.icon"
        >
          {{ link.label }}
        </UButton>
      </nav>
    </template>

    <template #right>
      <UColorModeButton />
    </template>
  </UHeader>
</template>
