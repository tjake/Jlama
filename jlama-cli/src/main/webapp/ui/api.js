
// Function to send a POST request to the API
function postRequest(input, session, signal) {
  const URL = `/chat/completions`;
  return fetch(URL, {
    method: 'POST',
    headers: {
      'X-Jlama-Session': session,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({"model": "jlama", "messages": [{"role": "user", "content": input}], "stream": true }),
    signal: signal
  });
}

// Function to stream the response from the server
async function getResponse(response, callback) {
  const reader = response.body.getReader();
  let partialLine = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    // Decode the received value and split by lines
    const textChunk = new TextDecoder().decode(value);
    const lines = (partialLine + textChunk).split('\n');
    partialLine = lines.pop(); // The last line might be incomplete

    for (const line of lines) {
      if (line.trim() === '') continue;
      if (line.startsWith('data:')) {
        const parsedResponse = JSON.parse(line.slice(5));
        callback(parsedResponse); // Process each response word
      } else {
        const parsedResponse = JSON.parse(line);
        callback(parsedResponse); // Process each response word
      }
    }
  }

  // Handle any remaining line
  if (partialLine.trim() !== '') {
    const parsedResponse = JSON.parse(partialLine);
    callback(parsedResponse);
  }
}