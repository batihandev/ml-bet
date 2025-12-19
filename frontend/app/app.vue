<script setup lang="ts">
const title = 'Football Edge Lab'
const description =
  'Train models and backtest football betting strategies with your ML pipeline.'

useHead({
  meta: [{ name: 'viewport', content: 'width=device-width, initial-scale=1' }],
  link: [{ rel: 'icon', href: '/favicon.ico' }],
  htmlAttrs: {
    lang: 'en'
  }
})

useSeoMeta({
  title,
  description,
  ogTitle: title,
  ogDescription: description,
  ogImage: '/og.png',
  twitterImage: '/og.png',
  twitterCard: 'summary_large_image'
})
</script>

<template>
  <UApp>
    <AppHeader />

    <UMain class="min-h-[calc(100vh-5rem)]">
      <NuxtPage />
    </UMain>

    <USeparator />
    <JobStatusBanner />
    <AppFooter />
  </UApp>
</template>
