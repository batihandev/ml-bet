<script setup lang="ts">
const year = new Date().getFullYear()
</script>

<template>
  <UFooter class="py-4">
    <template #left>
      <p class="text-xs text-muted">Football Edge Lab • {{ year }}</p>
    </template>

    <template #right>
      <div class="flex items-center gap-2">
        <span class="text-xs text-muted">Built with Nuxt &amp; Nuxt UI</span>
      </div>
    </template>
  </UFooter>
</template>
