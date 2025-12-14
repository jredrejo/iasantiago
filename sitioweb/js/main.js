// Gallery images array
const galleryImages = [
    { src: 'images/login_santiagoapostol.png', caption: 'Pantalla de login - IES Santiago Apóstol' },
    { src: 'images/login_buruaga.png', caption: 'Pantalla de login - IES Sáenz de Buruaga' },
    { src: 'images/funcionando1.png', caption: 'Sistema de IA funcionando' },
    { src: 'images/seleccion_temas.png', caption: 'Interfaz de selección de temas' },
    { src: 'images/IMG_20251212_141100.jpg', caption: 'Servidor donde está instalado todo el sistema' },
    { src: 'images/IMG_20251106_095547.jpg', caption: 'Alumnos de FP de Grado Medio en curso de formación - Vista 1' },
    { src: 'images/IMG_20251106_095613.jpg', caption: 'Alumnos de FP de Grado Medio en curso de formación - Vista 2' },
    { src: 'images/IMG_20251106_113823.jpg', caption: 'Alumnos de FP de Grado Superior en curso de formación - Vista 1' },
    { src: 'images/IMG_20251106_113836.jpg', caption: 'Alumnos de FP de Grado Superior en curso de formación - Vista 2' }

];

let currentImageIndex = 0;

// Get modal elements
const modal = document.getElementById('imageModal');
const modalImg = document.getElementById('modalImage');
const modalCaption = document.getElementById('modalCaption');

/**
 * Opens the modal with the specified image
 * @param {number} index - Index of the image in galleryImages array
 */
function openModal(index) {
    currentImageIndex = index;
    modal.classList.add('active');
    updateModalImage();
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
}

/**
 * Closes the modal
 */
function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
}

/**
 * Changes to the next or previous image
 * @param {number} direction - 1 for next, -1 for previous
 */
function changeImage(direction) {
    currentImageIndex += direction;

    // Wrap around if at the beginning or end
    if (currentImageIndex >= galleryImages.length) {
        currentImageIndex = 0;
    } else if (currentImageIndex < 0) {
        currentImageIndex = galleryImages.length - 1;
    }

    updateModalImage();
}

/**
 * Updates the modal with the current image
 */
function updateModalImage() {
    const image = galleryImages[currentImageIndex];
    modalImg.src = image.src;
    modalCaption.textContent = image.caption + ' (' + (currentImageIndex + 1) + '/' + galleryImages.length + ')';
}

// Close modal when clicking outside the image
modal.addEventListener('click', function(e) {
    if (e.target === modal) {
        closeModal();
    }
});

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (!modal.classList.contains('active')) return;

    switch(e.key) {
        case 'Escape':
            closeModal();
            break;
        case 'ArrowLeft':
            changeImage(-1);
            break;
        case 'ArrowRight':
            changeImage(1);
            break;
    }
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const headerHeight = document.querySelector('.header').offsetHeight;
            const targetPosition = target.offsetTop - headerHeight;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Add active class to nav links on scroll
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('.section, .hero');
    const navLinks = document.querySelectorAll('.nav-links a');
    const headerHeight = document.querySelector('.header').offsetHeight;

    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - headerHeight - 100;
        const sectionHeight = section.offsetHeight;

        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            current = section.getAttribute('id') || 'inicio';
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
});

// Touch swipe support for mobile
let touchStartX = 0;
let touchEndX = 0;

modal.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
}, false);

modal.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
}, false);

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) > swipeThreshold) {
        if (diff > 0) {
            // Swipe left - next image
            changeImage(1);
        } else {
            // Swipe right - previous image
            changeImage(-1);
        }
    }
}

// Preload images for smoother gallery experience
function preloadImages() {
    galleryImages.forEach(image => {
        const img = new Image();
        img.src = image.src;
    });
}

// Set data-alt attributes for gallery items from their img alt text
function setGalleryDataAlt() {
    const galleryItems = document.querySelectorAll('.gallery-item');
    galleryItems.forEach(item => {
        const img = item.querySelector('img');
        if (img && img.alt) {
            item.setAttribute('data-alt', img.alt);
        }
    });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    preloadImages();
    setGalleryDataAlt();
});
