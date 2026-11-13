// Uses Vite proxy so it works over any tunnel or domain automatically
const API_BASE_URL = "/api/v1";

export async function analyzeImage(imageBlob, question, sessionId = null) {
  const formData = new FormData();
  formData.append("question", question);

  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  if (imageBlob) {
    formData.append("file", imageBlob, "mobile_capture.jpg");
  }

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.message ||
        errorData.detail ||
        `Server returned error (${response.status})`,
    );
  }

  return response.json();
}
