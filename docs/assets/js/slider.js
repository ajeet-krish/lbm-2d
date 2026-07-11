// ==========================================================================
// LBM-2D: Comparison Slider (contour vs streamlines)
// Drag handle to reveal/hide the streamline overlay
// ==========================================================================

function initComparisonSlider(container) {
    const handle = container.querySelector('.slider-handle');
    const btn = container.querySelector('.slider-btn');
    const after = container.querySelector('.img-after');
    let dragging = false;

    function setPosition(pct) {
        pct = Math.max(0, Math.min(100, pct));
        if (after) after.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
        if (handle) handle.style.left = pct + '%';
    }

    function onMove(clientX) {
        const rect = container.getBoundingClientRect();
        const pct = ((clientX - rect.left) / rect.width) * 100;
        setPosition(pct);
    }

    container.addEventListener('mousedown', function(e) {
        dragging = true;
        onMove(e.clientX);
    });

    window.addEventListener('mousemove', function(e) {
        if (!dragging) return;
        onMove(e.clientX);
    });

    window.addEventListener('mouseup', function() {
        dragging = false;
    });

    // Touch support
    container.addEventListener('touchstart', function(e) {
        e.preventDefault();
        onMove(e.touches[0].clientX);
    }, { passive: false });

    container.addEventListener('touchmove', function(e) {
        e.preventDefault();
        onMove(e.touches[0].clientX);
    }, { passive: false });

    // Init at 50%
    setPosition(50);
}

// Auto-init all sliders on page load
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.comparison-slider').forEach(initComparisonSlider);
});
