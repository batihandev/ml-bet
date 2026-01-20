<template>
  <div class="space-y-12">
    <UBreadcrumb
      :items="[
        { label: 'Dashboard', to: '/' },
        { label: 'Settings' }
      ]"
    />

    <section class="space-y-6">
      <div class="flex items-center gap-2">
        <UIcon name="i-lucide-database" class="h-5 w-5 text-primary" />
        <h2 class="text-xl font-bold tracking-tight">Data Management</h2>
      </div>
      <DataStatus />
    </section>

    <USeparator />

    <section class="space-y-6">
      <div class="flex items-center gap-2">
        <UIcon name="i-lucide-brain" class="h-5 w-5 text-primary" />
        <h2 class="text-xl font-bold tracking-tight">Model Training</h2>
      </div>
      <TrainingPanel />
    </section>
  </div>
</template>
