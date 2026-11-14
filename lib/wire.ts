// The client wires and the three upstream wires, and the translations between them.
//
// The trial's client calls this entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent events.
// Each provider speaks its own wire: openai, anthropic or gemini. So a request may need
// translating in either direction, and the provider's answer back again:
//
//   client openai     -> provider openai     : pass through (model replaced)
//   client openai     -> provider anthropic  : openAIToAnthropicReq, then the answer back
//   client openai     -> provider gemini     : openAIToGeminiReq, then the answer back
//   client anthropic  -> provider openai     : anthropicToOpenAIReq, then the answer back
//   client anthropic  -> provider anthropic  : pass through (model replaced)
//   client anthropic  -> provider gemini     : anthropicToGeminiReq, then the answer back
//
// The translations must be faithful, because the provider echoes what it parsed and
// the client compares that echo with what it sent: the system text, the user text,
// the tool names in order, max_tokens, the stop sequences, the image and the JSON
// mode flag. A field rewritten or dropped loses the request. So every translator
// carries those fields across unchanged, in the shape the other wire spells them.
//
// The one asymmetry is JSON mode. An OpenAI request carries it as response_format;
// a Gemini request as generationConfig.responseMimeType; an Anthropic request has no
// such field at all (its echo always says json_mode false). So a JSON-mode OpenAI
// request only ever goes to an openai-wire or gemini-wire provider (the router
// enforces that), and a JSON-mode Anthropic request goes anywhere, with no JSON-mode
// signal added, because its echo does not carry one.
//
// The provider's answer is always plain text (the simulated service answers with an
// echo of the parsed request, never with a tool call), so the response translators
// handle text and, defensively, tool calls. Usage counts pass through as the wire
// reports them, and the stop reason maps end_turn to stop, max_tokens to length,
// tool_use to tool_calls and refusal to content_filter, per wire.
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same provider answer give the same bytes on a replay.
// The two model listings use fixed timestamps for the same reason: a `created` that
// moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
}

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: 'model',
      id: m.id,
      display_name: m.id,
      created_at: FIXED_ANTHROPIC_CREATED_AT,
    })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringListOf(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An OpenAI finish_reason as an Anthropic stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translators: client wire -> provider wire
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { type: 'text', text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON otherwise. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`:
 * the system prompt becomes a system message, image blocks become image_url data URLs,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages, tools and
 * tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through. Anthropic has no response_format, so none is added.
 */
export function anthropicToOpenAIReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') {
    return { type: 'text', text: part.text };
  }
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`:
 * system messages join into the system string, image_url parts become image blocks,
 * tool calls become tool_use blocks, tool messages become tool_result blocks, tools
 * and tool_choice are rewritten, and the output cap, temperature, top_p, stop
 * sequences and streaming pass through. A response_format is NOT carried: the
 * Anthropic wire has no field for it, and the router never sends a JSON-mode
 * OpenAI request to an Anthropic-wire provider.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    // user
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = openAIPartToBlock(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringListOf(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') {
        return [];
      }
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini content part; null for the rest. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') {
    return { text: part.text };
  }
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part; null for the rest. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') {
    return { text: block.text };
  }
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return {
        inlineData: {
          mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
          data: source.data,
        },
      };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { fileData: { fileUri: source.url } };
    }
  }
  return null;
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') {
      return [];
    }
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters)
          ? raw.function.parameters
          : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema)
          ? raw.input_schema
          : { type: 'object', properties: {} },
      },
    ];
  });
}

/** The generationConfig for a Gemini request: caps, stops, sampling, JSON mode. */
function geminiConfig(opts: {
  maxTokens?: number;
  stops?: string[];
  temperature?: number;
  topP?: number;
  jsonMode?: boolean;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The model
 * travels in the request path, not the body. System messages become the system
 * instruction, user parts become user contents, assistant turns become model
 * contents, tools become function declarations, and response_format becomes
 * responseMimeType. Tool messages have no Gemini shape that keeps the echo whole;
 * they arrive as functionResponse parts (the requests in this competition carry
 * none, being single-turn).
 */
export function openAIToGeminiReq(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: '',
              response: { result: toolResultText(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (Array.isArray(content)) {
      const parts: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const p = openAIPartToGemini(part);
        if (p !== null) parts.push(p);
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }

  const out: Json = { contents };
  if (systemParts.length > 0) {
    out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  }
  const declarations = openAIToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfig({
    maxTokens:
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : undefined,
    stops: stringListOf(body.stop),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The system string
 * becomes the system instruction, user blocks become user contents, assistant turns
 * become model contents, tool results become functionResponse parts, and tools
 * become function declarations. Anthropic has no JSON-mode signal, so none is set.
 */
export function anthropicToGeminiReq(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        parts.push({
          functionResponse: {
            name: '',
            response: { result: toolResultText(block.content) },
          },
        });
        continue;
      }
      const p = anthropicBlockToGemini(block);
      if (p !== null) parts.push(p);
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = anthropicToolsToGemini(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    stops: stringListOf(body.stop_sequences),
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translators: provider wire -> client wire
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropicResp(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: numberOrZero(usage.prompt_tokens),
      output_tokens: numberOrZero(usage.completion_tokens),
    },
  };
}

/** An Anthropic Messages object as a Chat Completions answer. */
export function anthropicToOpenAIResp(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: textParts.join(''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** The text of a Gemini answer's first candidate, '' when there is none. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return '';
  const content = isRecord(first.content) ? first.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer's usageMetadata as input/output token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    input: numberOrZero(meta.promptTokenCount),
    output: numberOrZero(meta.candidatesTokenCount),
  };
}

/** A Gemini generateContent answer as a Chat Completions answer. */
export function geminiToOpenAIResp(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(body) },
        finish_reason: geminiStopToOpenAI(first.finishReason),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicResp(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiStopToAnthropic(first.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events: the stream translators
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a chunk of SSE text into complete frames; the tail after the last blank line is left over. */
function splitFrames(buffer: string): { frames: SseFrame[]; rest: string } {
  const frames: SseFrame[] = [];
  let rest = buffer;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    const raw = rest.slice(0, at);
    rest = rest.slice(at + 2);
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
    at = rest.indexOf('\n\n');
  }
  return { frames, rest };
}

/** Copy a provider's event stream to the client as it arrives (same wire both ends). */
export async function pipeStream(
  stream: ReadableStream<Uint8Array>,
  write: (text: string) => void,
): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      write(Buffer.from(value).toString('utf8'));
    }
  } finally {
    reader.releaseLock();
  }
}

/** Read a whole SSE stream's frames, in order, as an async generator of parsed frames. */
async function* framesOf(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, rest } = splitFrames(buffer);
      buffer = rest;
      for (const frame of frames) yield frame;
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const { frames } = splitFrames(`${buffer}\n\n`);
      for (const frame of frames) yield frame;
    }
  } finally {
    reader.releaseLock();
  }
}

function jsonOf(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * An OpenAI event stream as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop. `model` is the
 * model the client asked for, named back to it.
 */
export async function* openAIToAnthropicStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of framesOf(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(pendingStop ?? 'end_turn');
      continue;
    }
    const chunk = jsonOf(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index: textIndex,
          content_block: { type: 'text', text: '' },
        });
      }
      yield sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: textIndex,
        delta: { type: 'text_delta', text: delta.content },
      });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: fn.arguments },
        });
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE]
    // have arrived: OpenAI sends the usage after the finish_reason, and Anthropic's
    // message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      pendingStop = openAIStopToAnthropic(choice.finish_reason);
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(pendingStop ?? 'end_turn');
  }
}

/** One OpenAI stream chunk: a content delta, a finish chunk, or a usage chunk. */
function openAIChunk(
  model: string,
  delta: Json,
  finish: string | null,
): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/**
 * An Anthropic event stream as OpenAI chunks: a role chunk, then content deltas,
 * then one finish chunk, then the usage chunk, then [DONE]. The usage chunk carries
 * the input tokens from message_start and the output tokens from message_delta.
 */
export async function* anthropicToOpenAIStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let roleSent = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolBlocks = new Map<number, { id: string; name: string }>();

  const finishAll = function* (): Generator<string> {
    if (finished) return;
    finished = true;
    yield sseEvent(null, openAIChunk(model, {}, stopReason ?? 'stop'));
    yield sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
    });
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of framesOf(upstream)) {
    if (frame.event === null) continue;
    const data = jsonOf(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) {
        inputTokens = numberOrZero(data.message.usage.input_tokens);
      }
      continue;
    }
    if (frame.event === 'content_block_start') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        toolBlocks.set(index, {
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
        });
      }
      if (!roleSent) {
        roleSent = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        if (!roleSent) {
          roleSent = true;
          yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
        }
        yield sseEvent(null, openAIChunk(model, { content: delta.text }, null));
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        const tool = toolBlocks.get(index);
        if (tool !== undefined) {
          yield sseEvent(null, {
            ...openAIChunk(model, {}, null),
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: tool.id,
                      type: 'function',
                      function: { name: tool.name, arguments: delta.partial_json },
                    },
                  ],
                },
                finish_reason: null,
              },
            ],
          });
        }
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        stopReason = anthropicStopToOpenAI(data.delta.stop_reason);
      }
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
    if (frame.event === 'message_stop') {
      yield* finishAll();
      continue;
    }
  }
  yield* finishAll();
}

/** The text pieces and the closing usage of one Gemini SSE frame, if it carries any. */
function geminiFrameParts(data: Json): { text: string; final: boolean; usage: { input: number; output: number } } {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const meta = isRecord(data.usageMetadata) ? data.usageMetadata : null;
  const final = typeof first.finishReason === 'string' || meta !== null;
  return {
    text,
    final,
    usage: {
      input: meta !== null ? numberOrZero(meta.promptTokenCount) : 0,
      output: meta !== null ? numberOrZero(meta.candidatesTokenCount) : 0,
    },
  };
}

/**
 * A Gemini event stream as OpenAI chunks: a role chunk, then content deltas, then
 * one finish chunk, then the usage chunk from the closing frame, then [DONE].
 */
export async function* geminiToOpenAIStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let roleSent = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;

  const finishAll = function* (finish: string): Generator<string> {
    if (finished) return;
    finished = true;
    yield sseEvent(null, openAIChunk(model, {}, finish));
    yield sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
    });
    yield 'data: [DONE]\n\n';
  };

  for await (const frame of framesOf(upstream)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    const { text, final, usage } = geminiFrameParts(data);
    if (text !== '') {
      if (!roleSent) {
        roleSent = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      yield sseEvent(null, openAIChunk(model, { content: text }, null));
    }
    if (final) {
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      inputTokens = usage.input;
      outputTokens = usage.output;
      yield* finishAll(geminiStopToOpenAI(first.finishReason));
    }
  }
  if (!finished) yield* finishAll('stop');
}

/**
 * A Gemini event stream as the Anthropic event sequence: message_start on the first
 * text, content_block_start/delta/stop for the one text block, then message_delta
 * with the stop reason and output tokens from the closing frame, then message_stop.
 * The Anthropic stream's usage check compares only the output tokens, so
 * message_start carries 0 input tokens (the Gemini stream only reports usage at
 * its close).
 */
export async function* geminiToAnthropicStream(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    ];
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of framesOf(upstream)) {
    const data = jsonOf(frame.data);
    if (data === null) continue;
    const { text, final, usage } = geminiFrameParts(data);
    if (text !== '') {
      yield* startEvents();
      yield sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text },
      });
    }
    if (final) {
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      outputTokens = usage.output;
      yield* startEvents();
      yield* finishEvents(geminiStopToAnthropic(first.finishReason));
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents('end_turn');
  }
}
