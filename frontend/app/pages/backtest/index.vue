<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <UBreadcrumb
        :items="[{ label: 'Dashboard', to: '/' }, { label: 'Backtests' }]"
      />

      <UButton
        to="/backtest/sweep"
        icon="i-lucide-grid-3x3"
        color="neutral"
        variant="ghost"
        size="sm"
      >
        Threshold Sweep
      </UButton>
    </div>

    <BacktestPanel />
  </div>
</template>
