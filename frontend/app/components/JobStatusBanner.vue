<script setup lang="ts">
const { isProcessing } = useJobEvents()
</script>

<template>
  <Transition name="fade">
    <div
      v-if="isProcessing"
      class="border-t border-gray-200/70 bg-primary-50/80 backdrop-blur-sm text-primary-900 dark:border-gray-800 dark:bg-primary-900/40 dark:text-primary-50"
    >
      <div
        class="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-2 text-xs"
      >
        <div class="flex items-center gap-2">
          <UIcon name="i-lucide-activity" class="h-4 w-4 animate-pulse" />
          <span>Running backtest… this may take a moment.</span>
        </div>

        <span class="hidden text-[10px] opacity-70 sm:inline">
          Live status via WebSocket
        </span>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease-out, transform 0.15s ease-out;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
