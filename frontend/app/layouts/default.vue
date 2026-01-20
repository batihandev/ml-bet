<template>
  <div class="flex flex-col min-h-screen">
    <AppHeader />
    <div class="flex flex-1">
      <AppSidebar />
      <UMain class="flex-1 p-6">
        <NuxtPage />
      </UMain>
    </div>
    <USeparator />
    <JobStatusBanner />
    <AppFooter />
  </div>
</template>

<style scoped>
:root {
  --header-height: 4rem;
}
</style>
