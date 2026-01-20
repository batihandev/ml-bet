<script setup lang="ts">
const heroTitle = 'Football Edge Lab'
const heroDescription =
  'Your advanced workspace for football betting analytics and machine learning.'
</script>

<template>
  <div class="space-y-12">
    <UPageHero
      :title="heroTitle"
      :description="heroDescription"
    />

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <UCard
        to="/backtest"
        class="transition-all hover:scale-[1.02] cursor-pointer"
        :ui="{ body: { padding: 'p-6' } }"
      >
        <div class="flex flex-col items-center text-center space-y-4">
          <div class="p-4 bg-emerald-500/10 rounded-full">
            <UIcon name="i-lucide-line-chart" class="h-8 w-8 text-emerald-500" />
          </div>
          <div>
            <h3 class="font-bold text-lg">Run Backtests</h3>
            <p class="text-sm text-muted">Test your betting strategies against historical data.</p>
          </div>
        </div>
      </UCard>

      <UCard
        to="/live"
        class="transition-all hover:scale-[1.02] cursor-pointer"
        :ui="{ body: { padding: 'p-6' } }"
      >
        <div class="flex flex-col items-center text-center space-y-4">
          <div class="p-4 bg-blue-500/10 rounded-full">
            <UIcon name="i-lucide-brain-circuit" class="h-8 w-8 text-blue-500" />
          </div>
          <div>
            <h3 class="font-bold text-lg">Live Predictions</h3>
            <p class="text-sm text-muted">Get real-time model predictions for upcoming matches.</p>
          </div>
        </div>
      </UCard>

      <UCard
        to="/settings"
        class="transition-all hover:scale-[1.02] cursor-pointer"
        :ui="{ body: { padding: 'p-6' } }"
      >
        <div class="flex flex-col items-center text-center space-y-4">
          <div class="p-4 bg-orange-500/10 rounded-full">
            <UIcon name="i-lucide-settings" class="h-8 w-8 text-orange-500" />
          </div>
          <div>
            <h3 class="font-bold text-lg">System Settings</h3>
            <p class="text-sm text-muted">Manage your datasets and retrain your models.</p>
          </div>
        </div>
      </UCard>
    </div>
  </div>
</template>
