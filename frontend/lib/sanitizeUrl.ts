/**
 * Validate that a URL uses a safe protocol (http, https, mailto).
 * Returns the URL if safe, or undefined if potentially dangerous
 * (e.g. javascript:, data:, vbscript:).
 */
export function sanitizeUrl(url: string | undefined | null): string | undefined {
  if (!url) return undefined;

  const trimmed = url.trim();
  if (!trimmed) return undefined;

  try {
    const parsed = new URL(trimmed);
    if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
      return trimmed;
    }
    return undefined;
  } catch {
    // Relative URLs are safe (resolve against page origin).
    // Reject anything that looks like a non-http protocol scheme.
    if (
      /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(trimmed) &&
      !trimmed.startsWith('http:') &&
      !trimmed.startsWith('https:') &&
      !trimmed.startsWith('mailto:')
    ) {
      return undefined;
    }
    return trimmed;
  }
}
