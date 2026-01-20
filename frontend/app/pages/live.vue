<template>
  <div class="space-y-6">
    <UBreadcrumb
      :items="[
        { label: 'Dashboard', to: '/' },
        { label: 'Live Predictions' }
      ]"
    />
    
    <LiveHistoryPanel />
  </div>
</template>
