<script setup lang="ts">
const heroTitle = 'Football Edge Lab'
const heroDescription =
  'Run training and FT 1X2 backtests on your football dataset. Quickly iterate on windows, edges, and staking while keeping everything reproducible.'
</script>

<template>
  <div>
    <UPageHero
      :title="heroTitle"
      :description="heroDescription"
      :links="[
        {
          label: 'Run training',
          to: '#models',
          icon: 'i-lucide-brain',
          size: 'lg'
        },
        {
          label: 'Run FT 1X2 backtest',
          to: '#backtests',
          trailingIcon: 'i-lucide-line-chart',
          size: 'lg',
          color: 'neutral',
          variant: 'subtle'
        }
      ]"
    />

    <UContainer content-class="pb-12">
      <DataStatus />
    </UContainer>

    <TrainingPanel />

    <BacktestPanel />

    <LiveHistoryPanel />
  </div>
</template>
