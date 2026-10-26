// =============================================
// LEAFCARE AI — COMPLETE FRONTEND SCRIPT
// =============================================

// IMPORTANT:
// Run the FastAPI backend before using predictions.
// If your website is deployed online, update API_URL
// to your deployed backend URL.

const API_URL = "https://leafcare-ai-project.onrender.com/predict";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp"
];


// =============================================
// DOM ELEMENTS
// =============================================

const $ = (id) => document.getElementById(id);

const imageInput = $("imageInput");
const cameraInput = $("cameraInput");

const uploadButton = $("uploadButton");
const cameraButton = $("cameraButton");
const uploadArea = $("uploadArea");

const cameraPanel = $("cameraPanel");
const cameraVideo = $("cameraVideo");
const cameraCanvas = $("cameraCanvas");
const captureButton = $("captureButton");
const closeCameraButton = $("closeCameraButton");
const cancelCameraButton = $("cancelCameraButton");

const previewPanel = $("previewPanel");
const imagePreview = $("imagePreview");
const fileInformation = $("fileInformation");
const removeImageButton = $("removeImageButton");

const predictButton = $("predictButton");
const loadingPanel = $("loadingPanel");

const errorMessage = $("errorMessage");
const errorText = $("errorText");

const resultPanel = $("resultPanel");
const predictedClass = $("predictedClass");
const confidenceValue = $("confidenceValue");
const confidenceProgress = $("confidenceProgress");
const confidenceFill = $("confidenceFill");

const topPredictionsSection = $("topPredictionsSection");
const topPredictions = $("topPredictions");
const analyzeAgainButton = $("analyzeAgainButton");

const menuToggle = $("menuToggle");
const navLinks = $("navLinks");


// =============================================
// APPLICATION STATE
// =============================================

let selectedFile = null;
let previewURL = null;
let cameraStream = null;
let isAnalyzing = false;


// =============================================
// FOOTER YEAR
// =============================================

const yearElement = $("currentYear");

if (yearElement) {
    yearElement.textContent = new Date().getFullYear();
}


// =============================================
// MOBILE NAVIGATION
// =============================================

if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
        const isOpen = navLinks.classList.toggle("open");

        menuToggle.setAttribute("aria-expanded", String(isOpen));
        menuToggle.textContent = isOpen ? "✕" : "☰";
    });

    navLinks.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", () => {
            navLinks.classList.remove("open");
            menuToggle.setAttribute("aria-expanded", "false");
            menuToggle.textContent = "☰";
        });
    });
}


// =============================================
// UPLOAD IMAGE
// =============================================

if (uploadButton && imageInput) {
    uploadButton.addEventListener("click", () => {
        if (!isAnalyzing) {
            imageInput.click();
        }
    });
}

if (imageInput) {
    imageInput.addEventListener("change", (event) => {
        const file = event.target.files?.[0];

        if (file) {
            handleSelectedFile(file);
        }

        // Allows selecting the same file again later.
        imageInput.value = "";
    });
}


// =============================================
// FILE VALIDATION AND PREVIEW
// =============================================

function handleSelectedFile(file) {
    if (isAnalyzing) return;

    hideError();

    if (!ALLOWED_TYPES.includes(file.type)) {
        showError("Please select a JPG, PNG or WEBP image.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showError("Image size must be 10 MB or less.");
        return;
    }

    if (file.size === 0) {
        showError("The selected image is empty. Please choose another image.");
        return;
    }

    stopCamera();
    releasePreviewURL();

    selectedFile = file;
    previewURL = URL.createObjectURL(file);

    imagePreview.src = previewURL;
    fileInformation.textContent =
        `${file.name} · ${formatFileSize(file.size)}`;

    previewPanel.hidden = false;
    uploadArea.hidden = true;
    cameraPanel.hidden = true;
    loadingPanel.hidden = true;
    resultPanel.hidden = true;

    predictButton.disabled = false;

    previewPanel.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


function releasePreviewURL() {
    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }
}


// =============================================
// DRAG AND DROP
// =============================================

if (uploadArea) {
    ["dragenter", "dragover"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, (event) => {
            event.preventDefault();

            if (!isAnalyzing) {
                uploadArea.classList.add("drag-over");
            }
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadArea.classList.remove("drag-over");
        });
    });

    uploadArea.addEventListener("drop", (event) => {
        if (isAnalyzing) return;

        const file = event.dataTransfer?.files?.[0];

        if (file) {
            handleSelectedFile(file);
        }
    });
}


// =============================================
// CAMERA
// =============================================

// On supported mobile browsers, prefer the device's
// rear camera through the native camera/file picker.
// On desktop browsers, try to open a live camera.

if (cameraButton) {
    cameraButton.addEventListener("click", async () => {
        if (isAnalyzing) return;

        hideError();

        const isMobile =
            /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

        if (isMobile) {
            // Mobile browsers can open the camera or photo chooser.
            // This path works without requiring live-camera APIs.
            if (cameraInput) {
                cameraInput.click();
            }
            return;
        }

        await openLiveCamera();
    });
}


if (cameraInput) {
    cameraInput.addEventListener("change", (event) => {
        const file = event.target.files?.[0];

        if (file) {
            handleSelectedFile(file);
        }

        cameraInput.value = "";
    });
}


async function openLiveCamera() {
    if (!navigator.mediaDevices ||
        typeof navigator.mediaDevices.getUserMedia !== "function") {
        showError(
            "Live camera is unavailable in this browser. Open the website on localhost or HTTPS, or use Upload Image."
        );
        return;
    }

    try {
        stopCamera();

        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: "environment" }
            }
        });

        cameraVideo.srcObject = cameraStream;

        cameraPanel.hidden = false;
        uploadArea.hidden = true;
        previewPanel.hidden = true;
        resultPanel.hidden = true;

        await cameraVideo.play();

        cameraPanel.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    } catch (error) {
        console.error("Camera error:", error);

        stopCamera();

        showError(
            "Could not open the camera. Allow camera permission and use localhost or HTTPS. You can also use Upload Image."
        );
    }
}


if (captureButton) {
    captureButton.addEventListener("click", () => {
        if (!cameraStream || !cameraVideo.videoWidth) {
            showError("The camera is not ready yet. Please wait a moment.");
            return;
        }

        const width = cameraVideo.videoWidth;
        const height = cameraVideo.videoHeight;

        cameraCanvas.width = width;
        cameraCanvas.height = height;

        const context = cameraCanvas.getContext("2d");

        if (!context) {
            showError("Unable to capture the image in this browser.");
            return;
        }

        context.drawImage(cameraVideo, 0, 0, width, height);

        cameraCanvas.toBlob((blob) => {
            if (!blob) {
                showError("Could not create the captured image. Please try again.");
                return;
            }

            const file = new File(
                [blob],
                `leaf-photo-${Date.now()}.jpg`,
                { type: "image/jpeg" }
            );

            stopCamera();
            handleSelectedFile(file);
        }, "image/jpeg", 0.92);
    });
}


function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }

    if (cameraVideo) {
        cameraVideo.pause();
        cameraVideo.srcObject = null;
    }

    if (cameraPanel) {
        cameraPanel.hidden = true;
    }
}


function closeCamera() {
    stopCamera();

    if (uploadArea && !selectedFile) {
        uploadArea.hidden = false;
    }
}


if (closeCameraButton) {
    closeCameraButton.addEventListener("click", closeCamera);
}

if (cancelCameraButton) {
    cancelCameraButton.addEventListener("click", closeCamera);
}


// =============================================
// REMOVE IMAGE / RESET
// =============================================

if (removeImageButton) {
    removeImageButton.addEventListener("click", resetDetector);
}

if (analyzeAgainButton) {
    analyzeAgainButton.addEventListener("click", resetDetector);
}


function resetDetector() {
    if (isAnalyzing) return;

    stopCamera();
    releasePreviewURL();

    selectedFile = null;

    if (imagePreview) {
        imagePreview.removeAttribute("src");
    }

    if (imageInput) imageInput.value = "";
    if (cameraInput) cameraInput.value = "";

    if (previewPanel) previewPanel.hidden = true;
    if (resultPanel) resultPanel.hidden = true;
    if (loadingPanel) loadingPanel.hidden = true;

    if (uploadArea) uploadArea.hidden = false;

    hideError();

    if (predictButton) {
        predictButton.disabled = false;
    }

    if (uploadArea) {
        uploadArea.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }
}


// =============================================
// ERROR DISPLAY
// =============================================

function showError(message) {
    if (!errorMessage || !errorText) {
        alert(message);
        return;
    }

    errorText.textContent = message;
    errorMessage.hidden = false;
}


function hideError() {
    if (errorMessage) {
        errorMessage.hidden = true;
    }
}


// =============================================
// API RESPONSE HELPERS
// =============================================

function getPredictionName(data) {
    return (
        data.predicted_class ??
        data.prediction ??
        data.disease ??
        data.class_name ??
        data.predicted_label ??
        data.label ??
        null
    );
}


function getConfidence(data) {
    return (
        data.confidence ??
        data.confidence_score ??
        data.probability ??
        data.score ??
        null
    );
}


function normalizeConfidence(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
        throw new Error(
            "The server did not return a valid confidence score."
        );
    }

    // Accept either 0.3174 or 31.74.
    const percent =
        number >= 0 && number <= 1
            ? number * 100
            : number;

    if (percent < 0 || percent > 100) {
        throw new Error("The server returned an invalid confidence score.");
    }

    return percent;
}


function prettifyClassName(name) {
    return String(name)
        .replace(/_/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}


// =============================================
// DISPLAY PREDICTION
// =============================================

function displayPrediction(data) {
    const disease = getPredictionName(data);

    if (typeof disease !== "string" || !disease.trim()) {
        throw new Error(
            "The API response did not contain a prediction name. Check the backend response."
        );
    }

    const confidence = normalizeConfidence(getConfidence(data));

    predictedClass.textContent = prettifyClassName(disease);
    confidenceValue.textContent = `${confidence.toFixed(2)}%`;

    confidenceFill.style.width = `${confidence}%`;

    confidenceProgress.setAttribute(
        "aria-valuenow",
        String(Math.round(confidence))
    );

    // Show alternative predictions only when the API returns them.
    const alternatives =
        data.top_predictions ??
        data.top_predictions_list ??
        data.alternatives ??
        data.predictions ??
        null;

    topPredictions.replaceChildren();

    const entries = normalizeAlternatives(alternatives, disease);

    if (entries.length > 0) {
        entries.forEach((item) => {
            const row = document.createElement("div");
            row.className = "top-prediction-item";

            const name = document.createElement("span");
            name.textContent = prettifyClassName(item.name);

            const score = document.createElement("strong");
            score.textContent = `${item.confidence.toFixed(2)}%`;

            row.append(name, score);
            topPredictions.appendChild(row);
        });

        topPredictionsSection.hidden = false;
    } else {
        topPredictionsSection.hidden = true;
    }

    resultPanel.hidden = false;

    resultPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


function normalizeAlternatives(value, mainDisease) {
    if (!Array.isArray(value)) return [];

    const results = [];

    for (const item of value) {
        let name;
        let score;

        if (typeof item === "string") {
            // A name alone has no confidence score to display.
            continue;
        }

        if (item && typeof item === "object") {
            name =
                item.class_name ??
                item.predicted_class ??
                item.label ??
                item.class ??
                item.name;

            score =
                item.confidence ??
                item.confidence_score ??
                item.probability ??
                item.score;
        }

        if (typeof name !== "string" || !name.trim()) continue;

        if (name === mainDisease) continue;

        try {
            const confidence = normalizeConfidence(score);

            results.push({
                name,
                confidence
            });
        } catch {
            // Ignore entries without a usable confidence score.
        }
    }

    return results.slice(0, 5);
}


// =============================================
// PREDICT DISEASE
// =============================================

if (predictButton) {
    predictButton.addEventListener("click", async () => {
        if (isAnalyzing) return;

        if (!selectedFile) {
            showError("Please upload or capture a leaf image first.");
            return;
        }

        hideError();

        isAnalyzing = true;

        predictButton.disabled = true;
        loadingPanel.hidden = false;
        resultPanel.hidden = true;

        try {
            const formData = new FormData();

            // FastAPI endpoint should accept UploadFile using field name "file".
            formData.append("file", selectedFile);

            const response = await fetch(API_URL, {
                method: "POST",
                body: formData
            });

            const responseText = await response.text();

            let data;

            try {
                data = JSON.parse(responseText);
            } catch {
                throw new Error(
                    "The backend returned an unreadable response. Check that the Python server is running."
                );
            }

            if (!response.ok) {
                const detail =
                    typeof data.detail === "string"
                        ? data.detail
                        : "The prediction request failed.";

                throw new Error(detail);
            }

            displayPrediction(data);

        } catch (error) {
            console.error("LeafCare AI error:", error);

            let message = error.message ||
                "Something went wrong while analyzing the image.";

            if (
                error instanceof TypeError ||
                /fetch/i.test(message)
            ) {
                message =
                    "Cannot connect to the AI backend. Please check your internet connection or try again in a moment.";
            }

            showError(message);

        } finally {
            isAnalyzing = false;
            loadingPanel.hidden = true;
            predictButton.disabled = !selectedFile;
        }
    });
}


// =============================================
// ACTIVE SECTION NAVIGATION
// =============================================

if ("IntersectionObserver" in window && navLinks) {
    const sections = document.querySelectorAll("main section[id]");

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;

                navLinks.querySelectorAll(".nav-link").forEach((link) => {
                    link.classList.toggle(
                        "active",
                        link.getAttribute("href") === `#${entry.target.id}`
                    );
                });
            });
        },
        {
            rootMargin: "-20% 0px -65% 0px"
        }
    );

    sections.forEach((section) => observer.observe(section));
}


// =============================================
// INITIALIZATION
// =============================================

window.addEventListener("beforeunload", stopCamera);

console.log(
    "LeafCare AI initialized. Prediction API:",
    API_URL
);
