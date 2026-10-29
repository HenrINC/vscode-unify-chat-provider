import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AuthTokenInfo } from '../../src/auth/types';
import type { ModelConfig, ProviderConfig } from '../../src/types';

const state = vi.hoisted(() => ({
  fetch: vi.fn<typeof fetch>(),
  releaseFetch: vi.fn<typeof fetch>(),
  createCustomFetch: vi.fn(),
  error: vi.fn(),
  getAvailableModels: vi.fn<(...args: unknown[]) => Promise<ModelConfig[]>>(),
}));

vi.mock('vscode', () => ({}));
vi.mock('../../src/logger', () => ({
  createSimpleHttpLogger: () => ({ error: state.error }),
}));
vi.mock('../../src/utils', () => ({
  DEFAULT_NORMAL_TIMEOUT_CONFIG: { connection: 10000, response: 30000 },
  resolveChatNetwork: (config: ProviderConfig) => ({ proxy: config.proxy }),
}));
vi.mock('../../src/client/utils', () => ({
  createCustomFetch: (options: unknown) => {
    state.createCustomFetch(options);
    return (input: Parameters<typeof fetch>[0], init: RequestInit) =>
      input === 'https://api.github.com/repos/openai/codex/releases/latest'
        ? state.releaseFetch(input, init)
        : state.fetch(input, init);
  },
  getToken: (credential: AuthTokenInfo) =>
    credential.kind === 'token' ? credential.token : undefined,
}));
vi.mock('../../src/client/openai/responses-client', () => ({
  OpenAIResponsesProvider: class {
    constructor(protected readonly config: ProviderConfig) {}

    protected buildHeaders(): Record<string, string> {
      return { ...this.config.extraHeaders };
    }

    async getAvailableModels(...args: unknown[]): Promise<ModelConfig[]> {
      return state.getAvailableModels(...args);
    }
  },
}));

import { OpenAICodexProvider } from '../../src/client/openai/codex-client';

const config: ProviderConfig = {
  type: 'openai-codex',
  name: 'Codex',
  baseUrl: 'https://chatgpt.com/backend-api/codex',
  auth: { method: 'openai-codex', bindingId: 'binding' },
  models: [],
  proxy: { type: 'direct' },
};
const credential: AuthTokenInfo = {
  kind: 'token',
  token: 'test-token',
  authContext: {
    method: 'openai-codex',
    bindingId: 'binding',
    sessionId: 'session',
    revision: 1,
    accountId: 'test-account',
  },
};
const catalogModel = {
  slug: 'gpt-6.1-sol',
  display_name: 'GPT-6.1-Sol',
  visibility: 'list',
  supported_in_api: true,
  context_window: 500000,
  max_context_window: 872000,
  input_modalities: ['text', 'image'],
  default_reasoning_level: 'medium',
  supported_reasoning_levels: [{ effort: 'medium' }, { effort: 'xhigh' }],
};

beforeEach(() => {
  vi.resetAllMocks();
  state.releaseFetch.mockImplementation(async () => Response.json({ tag_name: 'rust-v1.2.3' }));
});

describe('Codex model discovery', () => {
  it('never sends provider credentials or custom headers to the release API', async () => {
    state.fetch.mockResolvedValue(Response.json({ models: [] }));
    const controller = new AbortController();
    await new OpenAICodexProvider({
      ...config,
      extraHeaders: { 'X-Private-Key': 'private-value', Authorization: 'private-auth' },
    }).getAvailableModels(credential, undefined, controller.signal);
    expect(state.releaseFetch).toHaveBeenCalledExactlyOnceWith(
      'https://api.github.com/repos/openai/codex/releases/latest',
      {
        method: 'GET',
        headers: {
          Accept: 'application/vnd.github+json',
          'User-Agent': 'unify-chat-provider',
        },
        signal: controller.signal,
        redirect: 'error',
      },
    );
  });

  it('resolves the latest version on every model refresh', async () => {
    state.fetch.mockImplementation(async () => Response.json({ models: [] }));
    const provider = new OpenAICodexProvider(config);
    await provider.getAvailableModels(credential);
    expect(state.releaseFetch).toHaveBeenCalledOnce();

    state.releaseFetch.mockResolvedValueOnce(Response.json({ tag_name: 'rust-v1.3.0' }));
    await provider.getAvailableModels(credential);
    expect(state.releaseFetch).toHaveBeenCalledTimes(2);
    expect(state.fetch).toHaveBeenLastCalledWith(
      'https://chatgpt.com/backend-api/codex/models?client_version=1.3.0',
      expect.objectContaining({ headers: expect.objectContaining({
        'User-Agent': 'codex-tui/1.3.0',
      }) }),
    );
  });

  it.each([403, 429, 503])('stops discovery on release lookup failure and allows retry: HTTP %s', async (status) => {
    state.releaseFetch.mockResolvedValueOnce(new Response(null, { status }));
    const provider = new OpenAICodexProvider(config);
    await expect(provider.getAvailableModels(credential))
      .rejects.toThrow(`Failed to get latest Codex version: HTTP ${status}`);
    expect(state.fetch).not.toHaveBeenCalled();

    state.fetch.mockResolvedValue(Response.json({ models: [] }));
    await expect(provider.getAvailableModels(credential)).resolves.toEqual([]);
    expect(state.releaseFetch).toHaveBeenCalledTimes(2);
  });

  it.each([
    null,
    {},
    { tag_name: 'v1.2.3' },
    { tag_name: 'rust-v1.2.3-alpha.1' },
    { tag_name: 'rust-v1.2.3', prerelease: true },
    { tag_name: 'rust-v1.2.3', draft: true },
  ])('rejects malformed or nonstable releases: %j', async (release) => {
    state.releaseFetch.mockResolvedValueOnce(Response.json(release));
    await expect(new OpenAICodexProvider(config).getAvailableModels(credential))
      .rejects.toThrow('unexpected release');
    expect(state.fetch).not.toHaveBeenCalled();
  });

  it('does not reuse a previous version if the next lookup fails', async () => {
    state.fetch.mockResolvedValueOnce(Response.json({ models: [] }));
    const provider = new OpenAICodexProvider(config);
    await provider.getAvailableModels(credential);
    state.releaseFetch.mockRejectedValueOnce(new Error('Release lookup unavailable'));
    await expect(provider.getAvailableModels(credential))
      .rejects.toThrow('Release lookup unavailable');
    expect(state.fetch).toHaveBeenCalledOnce();
  });

  it('fetches the authenticated catalogue and maps visible model metadata', async () => {
    state.fetch.mockResolvedValue(Response.json({ models: [
      catalogModel,
      { ...catalogModel, slug: 'hidden', visibility: 'hide' },
      { ...catalogModel, slug: 'unsupported', supported_in_api: false },
      { ...catalogModel, slug: '' },
      null,
    ] }));
    const controller = new AbortController();
    const models = await new OpenAICodexProvider(config).getAvailableModels(
      credential, undefined, controller.signal,
    );

    expect(models).toEqual([{
      id: 'gpt-6.1-sol',
      name: 'GPT-6.1-Sol',
      maxInputTokens: 500000,
      maxOutputTokens: undefined,
      thinking: { type: 'enabled', effort: 'xhigh', summary: 'auto' },
      capabilities: { toolCalling: true, imageInput: true },
    }]);
    expect(state.fetch).toHaveBeenCalledWith(
      'https://chatgpt.com/backend-api/codex/models?client_version=1.2.3',
      expect.objectContaining({
        method: 'GET',
        signal: controller.signal,
        redirect: 'error',
        headers: expect.objectContaining({
          Authorization: 'Bearer test-token',
          Accept: 'application/json',
          'Chatgpt-Account-Id': 'test-account',
          Originator: 'codex-tui',
          'User-Agent': 'codex-tui/1.2.3',
        }),
      }),
    );
    expect(state.createCustomFetch).toHaveBeenCalledWith(expect.objectContaining({
      proxy: config.proxy,
      connectionTimeoutMs: 10000,
      responseTimeoutMs: 30000,
      type: 'normal',
    }));
    expect(state.getAvailableModels).not.toHaveBeenCalled();
  });

  it('uses a supported default effort when xhigh is unavailable', async () => {
    state.fetch.mockResolvedValue(Response.json({ models: [{
      ...catalogModel,
      supported_reasoning_levels: [{ effort: 'low' }, { effort: 'medium' }],
      supports_reasoning_summaries: false,
      context_window: -1,
      input_modalities: ['text'],
    }] }));
    const [model] = await new OpenAICodexProvider(config).getAvailableModels(credential);
    expect(model?.thinking).toEqual({ type: 'enabled', effort: 'medium', summary: undefined });
    expect(model?.maxInputTokens).toBeUndefined();
    expect(model?.capabilities?.imageInput).toBe(false);
  });

  it('does not invent an effort or context limit when metadata is absent', async () => {
    state.fetch.mockResolvedValue(Response.json({ models: [{
      slug: 'future-model', visibility: 'list', supported_in_api: true,
    }] }));
    const [model] = await new OpenAICodexProvider(config).getAvailableModels(credential);
    expect(model?.id).toBe('future-model');
    expect(model?.thinking?.effort).toBeUndefined();
    expect(model?.maxInputTokens).toBeUndefined();
  });

  it('delegates API-key configurations without contacting ChatGPT', async () => {
    const apiCredential: AuthTokenInfo = { kind: 'token', token: 'api-key' };
    const refresh = vi.fn(async () => apiCredential);
    const controller = new AbortController();
    state.getAvailableModels.mockResolvedValue([{ id: 'proxy-model' }]);
    const provider = new OpenAICodexProvider({
      ...config, baseUrl: 'https://example.com/v1', auth: { method: 'api-key' },
    });
    await expect(provider.getAvailableModels(apiCredential, refresh, controller.signal))
      .resolves.toEqual([{ id: 'proxy-model' }]);
    expect(state.getAvailableModels).toHaveBeenCalledWith(apiCredential, refresh, controller.signal);
    expect(state.fetch).not.toHaveBeenCalled();
    expect(state.releaseFetch).not.toHaveBeenCalled();
  });

  it('rejects missing authentication before making a request', async () => {
    await expect(new OpenAICodexProvider(config).getAvailableModels({ kind: 'none' }))
      .rejects.toThrow('Authentication required');
    expect(state.fetch).not.toHaveBeenCalled();
    expect(state.releaseFetch).not.toHaveBeenCalled();
  });

  it.each([401, 403, 429, 500])('propagates HTTP %s without a static replacement', async (status) => {
    state.fetch.mockResolvedValue(new Response(null, { status }));
    await expect(new OpenAICodexProvider(config).getAvailableModels(credential))
      .rejects.toThrow(`HTTP ${status}`);
    expect(state.error).toHaveBeenCalledOnce();
  });

  it.each([null, {}, { models: {} }])('rejects malformed catalogue envelopes: %j', async (body) => {
    state.fetch.mockResolvedValue(Response.json(body));
    await expect(new OpenAICodexProvider(config).getAvailableModels(credential))
      .rejects.toThrow('unexpected Codex response');
  });

  it('returns a valid empty catalogue without injecting static models', async () => {
    state.fetch.mockResolvedValue(Response.json({ models: [] }));
    await expect(new OpenAICodexProvider(config).getAvailableModels(credential))
      .resolves.toEqual([]);
  });

  it('propagates cancellation', async () => {
    const controller = new AbortController();
    controller.abort();
    state.fetch.mockRejectedValue(controller.signal.reason);
    await expect(new OpenAICodexProvider(config).getAvailableModels(
      credential, undefined, controller.signal,
    )).rejects.toBe(controller.signal.reason);
    expect(state.releaseFetch).not.toHaveBeenCalled();
    expect(state.fetch).not.toHaveBeenCalled();
  });
});