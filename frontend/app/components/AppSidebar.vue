<script setup lang="ts">
const links = [
  [
    {
      label: 'Dashboard',
      icon: 'i-lucide-home',
      to: '/'
    },
    {
      label: 'Backtests',
      icon: 'i-lucide-line-chart',
      to: '/backtest'
    },
    {
      label: 'Live Predictions',
      icon: 'i-lucide-brain-circuit',
      to: '/live'
    }
  ],
  [
    {
      label: 'Settings',
      icon: 'i-lucide-settings',
      to: '/settings'
    }
  ]
]
</script>

<template>
  <aside class="w-64 border-r border-gray-200/60 dark:border-gray-800/80 p-4 sticky top-[var(--ui-header-height)] h-[calc(100vh-var(--ui-header-height))] overflow-y-auto hidden md:block">
    <UNavigationMenu
      :items="links"
      orientation="vertical"
      class="w-full"
    />
  </aside>
</template>
