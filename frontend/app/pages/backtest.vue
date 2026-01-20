<template>
  <div class="space-y-6">
    <UBreadcrumb
      :items="[
        { label: 'Dashboard', to: '/' },
        { label: 'Backtests' }
      ]"
    />
    
    <BacktestPanel />
  </div>
</template>
