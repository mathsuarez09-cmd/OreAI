(function (Scratch) {
    'use strict';

    if (!Scratch.extensions.unsandboxed) {
        throw new Error('OreAI must be run unsandboxed to perform network requests.');
    }

    class OreAI {
        constructor() {
            // Configuration State
            this.apiKey = '';
            this.provider = 'Google Gemini';
            this.model = 'gemini-3.5-flash';
            this.systemPrompt = 'You are a helpful AI assistant integrated into a Scratch project.';
            this.temperature = 0.7;
            this.maxTokens = 500;

            // Chat Memory State
            this.chatHistory = [];
            this.chatMemoryLimit = 10;

            // Image Generation State
            this.lastGeneratedImageUrl = '';

            // Debug & Status
            this.lastError = '';
            this.lastResponseTimeMs = 0;
            this.status = 'Idle';
        }

        getInfo() {
            return {
                id: 'oreAI',
                name: 'OreAI v2.0',
                color1: '#7b2cbf',
                color2: '#5a189a',
                color3: '#3c096c',
                blocks: [
                    // --- ⚙️ CATEGORY 1: CONFIGURATION MANAGEMENT ---
                    '---',
                    '⚙️ Configuration Management',
                    {
                        opcode: 'setApiKey',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set API key to [KEY]',
                        arguments: {
                            KEY: { type: Scratch.ArgumentType.STRING, defaultValue: 'AIzaSy...' }
                        }
                    },
                    {
                        opcode: 'setProvider',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set AI provider to [PROVIDER]',
                        arguments: {
                            PROVIDER: { type: Scratch.ArgumentType.STRING, menu: 'providers', defaultValue: 'Google Gemini' }
                        }
                    },
                    {
                        opcode: 'setModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set model to [MODEL]',
                        arguments: {
                            MODEL: { type: Scratch.ArgumentType.STRING, menu: 'models', defaultValue: 'gemini-3.5-flash' }
                        }
                    },
                    {
                        opcode: 'setCustomModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set custom model ID to [MODEL_ID]',
                        arguments: {
                            MODEL_ID: { type: Scratch.ArgumentType.STRING, defaultValue: 'gemini-3.5-flash' }
                        }
                    },
                    {
                        opcode: 'setTemperature',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set creativity (temperature) to [TEMP]',
                        arguments: {
                            TEMP: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.7 }
                        }
                    },
                    {
                        opcode: 'setMaxTokens',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set max response tokens to [TOKENS]',
                        arguments: {
                            TOKENS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 500 }
                        }
                    },

                    // --- 🎭 CATEGORY 2: PERSONALITY MANAGEMENT ---
                    '---',
                    '🎭 Personality Management',
                    {
                        opcode: 'setSystemPrompt',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set custom system prompt to [PROMPT]',
                        arguments: {
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'You are a helpful assistant.' }
                        }
                    },
                    {
                        opcode: 'setPersonalityPreset',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'apply personality preset [PRESET]',
                        arguments: {
                            PRESET: { type: Scratch.ArgumentType.STRING, menu: 'personalityPresets', defaultValue: 'Friendly NPC' }
                        }
                    },
                    {
                        opcode: 'getSystemPrompt',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current system prompt',
                        disableMonitor: true
                    },

                    // --- 💬 CATEGORY 3: CHATBOT MANAGEMENT ---
                    '---',
                    '💬 Chatbot Management',
                    {
                        opcode: 'generateResponse',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'prompt AI [PROMPT]',
                        arguments: {
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'Hello! Tell me a fun fact.' }
                        }
                    },
                    {
                        opcode: 'sendChatMessage',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'chat with AI memory [PROMPT]',
                        arguments: {
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'What was my last question?' }
                        }
                    },
                    {
                        opcode: 'clearChatHistory',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'clear chat memory'
                    },
                    {
                        opcode: 'setChatMemoryLimit',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set chat memory limit to [LIMIT] turns',
                        arguments: {
                            LIMIT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 10 }
                        }
                    },
                    {
                        opcode: 'getChatHistoryLength',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'chat memory message count'
                    },

                    // --- 🖼️ CATEGORY 4: IMAGE GENERATION MANAGEMENT ---
                    '---',
                    '🖼️ Image Generation Management',
                    {
                        opcode: 'generateImage',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'generate image from prompt [PROMPT]',
                        arguments: {
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'A futuristic floating city pixel art' }
                        }
                    },
                    {
                        opcode: 'getLastImageUrl',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last generated image URL',
                        disableMonitor: true
                    },

                    // --- 🛠️ CATEGORY 5: MISCELLANEOUS & DEBUGGING ---
                    '---',
                    '🛠️ Miscellaneous',
                    {
                        opcode: 'getLastError',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last error',
                        disableMonitor: true
                    },
                    {
                        opcode: 'getLatency',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last call latency (ms)'
                    },
                    {
                        opcode: 'getStatus',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'AI status'
                    },
                    {
                        opcode: 'isReady',
                        blockType: Scratch.BlockType.BOOLEAN,
                        text: 'is API key set?'
                    }
                ],
                menus: {
                    providers: {
                        acceptReporters: true,
                        items: ['Google Gemini', 'OpenAI']
                    },
                    models: {
                        acceptReporters: true,
                        items: [
                            'gemini-3.5-flash',
                            'gemini-3.6-flash',
                            'gemini-3.1-flash-lite',
                            'gpt-5.4-mini',
                            'gpt-5.4'
                        ]
                    },
                    personalityPresets: {
                        acceptReporters: true,
                        items: [
                            'Friendly NPC',
                            'Grumpy Shopkeeper',
                            'Wise Mentor',
                            'Sci-Fi AI Assistant',
                            'Concise Assistant (Max 15 words)'
                        ]
                    }
                }
            };
        }

        // ==========================================
        // ⚙️ CONFIGURATION MANAGEMENT
        // ==========================================
        setApiKey(args) {
            this.apiKey = Scratch.Cast.toString(args.KEY).trim();
        }

        setProvider(args) {
            this.provider = Scratch.Cast.toString(args.PROVIDER);
        }

        setModel(args) {
            this.model = Scratch.Cast.toString(args.MODEL);
        }

        setCustomModel(args) {
            this.model = Scratch.Cast.toString(args.MODEL_ID).trim();
        }

        setTemperature(args) {
            let temp = Scratch.Cast.toNumber(args.TEMP);
            this.temperature = Math.max(0, Math.min(2, temp));
        }

        setMaxTokens(args) {
            let tokens = Scratch.Cast.toNumber(args.TOKENS);
            this.maxTokens = Math.max(1, tokens);
        }

        // ==========================================
        // 🎭 PERSONALITY MANAGEMENT
        // ==========================================
        setSystemPrompt(args) {
            this.systemPrompt = Scratch.Cast.toString(args.PROMPT);
        }

        setPersonalityPreset(args) {
            const preset = Scratch.Cast.toString(args.PRESET);
            switch (preset) {
                case 'Friendly NPC':
                    this.systemPrompt = 'You are a cheerful and helpful NPC in a fantasy game world.';
                    break;
                case 'Grumpy Shopkeeper':
                    this.systemPrompt = 'You are a grumpy shopkeeper. You sell goods but complain about prices and customers constantly. Keep responses blunt and brief.';
                    break;
                case 'Wise Mentor':
                    this.systemPrompt = 'You are an ancient, wise wizard offering mystical guidance and advice to a young hero.';
                    break;
                case 'Sci-Fi AI Assistant':
                    this.systemPrompt = 'You are an advanced futuristic spaceship AI. Speak efficiently with tactical terminology.';
                    break;
                case 'Concise Assistant (Max 15 words)':
                    this.systemPrompt = 'You are a helpful assistant. Crucial rule: Never respond with more than 15 words total.';
                    break;
                default:
                    this.systemPrompt = 'You are a helpful assistant.';
            }
        }

        getSystemPrompt() {
            return this.systemPrompt;
        }

        // ==========================================
        // 💬 CHATBOT MANAGEMENT
        // ==========================================
        clearChatHistory() {
            this.chatHistory = [];
        }

        setChatMemoryLimit(args) {
            this.chatMemoryLimit = Math.max(2, Scratch.Cast.toNumber(args.LIMIT));
        }

        getChatHistoryLength() {
            return this.chatHistory.length;
        }

        // Single isolated prompt (Original Block)
        async generateResponse(args) {
            const prompt = Scratch.Cast.toString(args.PROMPT);
            return await this._executeRequest(prompt, false);
        }

        // Chat prompt with full conversation history
        async sendChatMessage(args) {
            const prompt = Scratch.Cast.toString(args.PROMPT);
            return await this._executeRequest(prompt, true);
        }

        // ==========================================
        // 🖼️ IMAGE GENERATION MANAGEMENT
        // ==========================================
        async generateImage(args) {
            const prompt = Scratch.Cast.toString(args.PROMPT);
            if (!this.apiKey) {
                this.lastError = 'API Key is missing for Image Generation.';
                return;
            }

            this.status = 'Generating Image...';
            const startTime = Date.now();

            try {
                if (this.provider === 'OpenAI') {
                    const response = await Scratch.fetch('https://api.openai.com/v1/images/generations', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${this.apiKey}`
                        },
                        body: JSON.stringify({
                            model: 'dall-e-3',
                            prompt: prompt,
                            n: 1,
                            size: '1024x1024'
                        })
                    });

                    if (!response.ok) {
                        const errData = await response.json().catch(() => ({}));
                        throw new Error(errData.error?.message || `HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    this.lastGeneratedImageUrl = data.data?.[0]?.url || '';
                } else {
                    // Gemini / Google Imagen call
                    const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${this.apiKey}`;
                    const response = await Scratch.fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            instances: [{ prompt: prompt }],
                            parameters: { sampleCount: 1 }
                        })
                    });

                    if (!response.ok) {
                        const errData = await response.json().catch(() => ({}));
                        throw new Error(errData.error?.message || `HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    const mimeType = data.predictions?.[0]?.mimeType || 'image/png';
                    const base64 = data.predictions?.[0]?.bytesBase64Encoded || '';
                    this.lastGeneratedImageUrl = `data:${mimeType};base64,${base64}`;
                }

                this.status = 'Idle';
                this.lastResponseTimeMs = Date.now() - startTime;
            } catch (err) {
                this.lastError = err.message || String(err);
                this.status = 'Error';
                this.lastGeneratedImageUrl = '';
            }
        }

        getLastImageUrl() {
            return this.lastGeneratedImageUrl;
        }

        // ==========================================
        // 🛠️ MISCELLANEOUS & UTILITIES
        // ==========================================
        getLastError() {
            return this.lastError;
        }

        getLatency() {
            return this.lastResponseTimeMs;
        }

        getStatus() {
            return this.status;
        }

        isReady() {
            return this.apiKey.length > 0;
        }

        // ==========================================
        // 🔒 PRIVATE NETWORK PIPELINE
        // ==========================================
        async _executeRequest(userPrompt, useMemory) {
            if (!this.apiKey) {
                this.lastError = 'API Key is missing.';
                return 'Error: API Key is missing.';
            }

            this.status = 'Thinking...';
            const startTime = Date.now();

            try {
                let responseText = '';
                if (this.provider === 'Google Gemini') {
                    responseText = await this._callGemini(userPrompt, useMemory);
                } else {
                    responseText = await this._callOpenAI(userPrompt, useMemory);
                }

                if (useMemory) {
                    this.chatHistory.push({ role: 'user', content: userPrompt });
                    this.chatHistory.push({ role: 'assistant', content: responseText });

                    // Prune history to limit size
                    while (this.chatHistory.length > this.chatMemoryLimit * 2) {
                        this.chatHistory.shift();
                    }
                }

                this.lastResponseTimeMs = Date.now() - startTime;
                this.status = 'Idle';
                return responseText;
            } catch (err) {
                this.lastError = err.message || String(err);
                this.status = 'Error';
                return `Error: ${this.lastError}`;
            }
        }

        async _callOpenAI(prompt, useMemory) {
            const url = 'https://api.openai.com/v1/chat/completions';
            
            let messages = [{ role: 'system', content: this.systemPrompt }];
            if (useMemory) {
                messages = messages.concat(this.chatHistory);
            }
            messages.push({ role: 'user', content: prompt });

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    model: this.model,
                    temperature: this.temperature,
                    max_tokens: this.maxTokens,
                    messages: messages
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error?.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            return data.choices?.[0]?.message?.content || '';
        }

        async _callGemini(prompt, useMemory) {
            const endpointModel = this.model.includes('gemini') ? this.model : 'gemini-3.5-flash';
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${endpointModel}:generateContent?key=${this.apiKey}`;

            let contents = [];
            if (useMemory) {
                contents = this.chatHistory.map(item => ({
                    role: item.role === 'assistant' ? 'model' : 'user',
                    parts: [{ text: item.content }]
                }));
            }
            contents.push({ role: 'user', parts: [{ text: prompt }] });

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    systemInstruction: { parts: [{ text: this.systemPrompt }] },
                    generationConfig: {
                        temperature: this.temperature,
                        maxOutputTokens: this.maxTokens
                    },
                    contents: contents
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error?.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
        }
    }

    Scratch.extensions.register(new OreAI());
})(Scratch);
