const copy = document.querySelector('#copy');
copy.addEventListener('click', async () => {
  const citation = document.querySelector('#citation code');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copy.textContent = 'Copied';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    copy.textContent = 'Citation selected';
  }
});

for (const list of document.querySelectorAll('.house-tabs')) {
  const tabs = [...list.querySelectorAll('[role=tab]')];
  function showTab(selected) {
    for (const tab of tabs) {
      const active = tab === selected;
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      panel.hidden = !active;
      tab.tabIndex = active ? 0 : -1;
      tab.setAttribute('aria-selected', String(active));
    }
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => showTab(tab));
    tab.addEventListener('keydown', event => {
      const targets = {
        ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1
      };
      if (!(event.key in targets)) return;
      event.preventDefault();
      const next = tabs[targets[event.key]];
      next.focus();
      showTab(next);
    });
  });
}

// Only visible comparisons play automatically. Remember pauses made with the native controls.
const manuallyPaused = new WeakSet();
const viewportPauses = new WeakSet();
const galleryObserver = new IntersectionObserver(entries => {
  for (const { target: video, isIntersecting } of entries) {
    if (isIntersecting && !video.closest('[hidden]')) {
      if (!manuallyPaused.has(video)) video.play().catch(() => {});
    } else if (!video.paused) {
      viewportPauses.add(video);
      video.pause();
    }
  }
});
for (const video of document.querySelectorAll('#demos .comparison video')) {
  video.addEventListener('pause', () => {
    if (!viewportPauses.delete(video)) manuallyPaused.add(video);
  });
  video.addEventListener('play', () => manuallyPaused.delete(video));
  galleryObserver.observe(video);
}

// Give a downward wheel gesture at the footer a small, reversible overshoot.
const scrollArea = document.querySelector('.page-scroll');
const pageMain = scrollArea.querySelector('main');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let pull = 0;
let releaseTimer;
function releaseFooter() {
  pull = 0;
  pageMain.style.transform = '';
  pageMain.classList.remove('pulling-footer');
}
scrollArea.addEventListener('wheel', event => {
  if (event.ctrlKey || reducedMotion.matches || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
  const remaining = scrollArea.scrollHeight - scrollArea.clientHeight - scrollArea.scrollTop;
  if (remaining > 1 || event.deltaY <= 0) {
    if (pull) releaseFooter();
    return;
  }
  const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? scrollArea.clientHeight : 1;
  pull += event.deltaY * units;
  const offset = 30 * (1 - Math.exp(-pull / 180));
  pageMain.classList.add('pulling-footer');
  pageMain.style.transform = 'translateY(-' + offset + 'px)';
  clearTimeout(releaseTimer);
  releaseTimer = setTimeout(releaseFooter, 85);
}, { passive: true });
