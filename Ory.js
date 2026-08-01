(function (Scratch) {
    'use strict';

    if (!Scratch.extensions.unsandboxed) {
        throw new Error('OreAI must be run unsandboxed to perform network requests.');
    }

    // ==========================================
    // 🧠 Sux Class (Legacy / Made by me)
    // ==========================================
    class Sux {
        constructor(embeddingDim = 16, numLayers = 2, numHeads = 2) {
            this.tokenToId = new Map();
            this.idToToken = new Map();
            this.bigramCounts = new Map();
            this.embeddingDim = embeddingDim;
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.embeddings = new Map();
            this.parameters = {};
            this.layerNames = ["embedding", "self_attention", "feed_forward", "output"];
            this.postTrainingHistory = [];
            this.instructionTuningExamples = [];
            this.seed = 42;
        }

        _random() {
            let x = Math.sin(this.seed++) * 10000;
            return x - Math.floor(x);
        }

        _randomEmbedding() {
            let vector = [];
            for (let i = 0; i < this.embeddingDim; i++) {
                vector.push((this._random() * 0.2) - 0.1);
            }
            return vector;
        }

        tokenize(text) {
            const tokens = text.trim().split(/\s+/).filter(t => t.length > 0);
            const tokenIds = [];
            for (const token of tokens) {
                if (!this.tokenToId.has(token)) {
                    const idx = this.tokenToId.size;
                    this.tokenToId.set(token, idx);
                    this.idToToken.set(idx, token);
                    this.embeddings.set(idx, this._randomEmbedding());
                }
                tokenIds.push(this.tokenToId.get(token));
            }
            this._refreshParameters();
            return tokenIds;
        }

        detokenize(tokenIds) {
            return tokenIds.map(t => this.idToToken.get(t) || "").join(" ").trim();
        }

        _refreshParameters() {
            const tokenEmbeddingParams = this.embeddings.size * this.embeddingDim;
            const attentionParams = this.embeddingDim * this.embeddingDim * this.numHeads;
            const feedForwardParams = this.embeddingDim * this.embeddingDim;
            const outputParams = Math.max(this.tokenToId.size, 1) * this.embeddingDim;
            this.parameters = {
                embedding: tokenEmbeddingParams,
                self_attention: attentionParams,
                feed_forward: feedForwardParams,
                output: outputParams,
                total: tokenEmbeddingParams + attentionParams + feedForwardParams + outputParams
            };
        }

        _embeddingFor(tokenId) {
            return this.embeddings.get(tokenId) || new Array(this.embeddingDim).fill(0.0);
        }

        selfAttention(tokenIds) {
            if (!tokenIds.length) return new Array(this.embeddingDim).fill(0.0);
            const vectors = tokenIds.map(id => this._embeddingFor(id));
            const context = [];
            for (let dim = 0; dim < this.embeddingDim; dim++) {
                let total = 0.0;
                vectors.forEach((vector, index) => {
                    total += vector[dim] * (index + 1);
                });
                context.push(total / Math.max(vectors.length, 1));
            }
            return context;
        }

        transformerBlock(tokenIds) {
            const attentionOutput = this.selfAttention(tokenIds);
            return attentionOutput.map((val, index) => val + (index + 1) * 0.01);
        }

        deepNeuralNetworkLayers(tokenIds) {
            return {
                embedding: tokenIds.map(id => this._embeddingFor(id)),
                self_attention: this.selfAttention(tokenIds),
                transformer: this.transformerBlock(tokenIds),
                layers: this.layerNames
            };
        }

        train(corpus) {
            this.tokenToId.clear();
            this.idToToken.clear();
            this.bigramCounts.clear();
            this.embeddings.clear();
            this.seed = 42;

            const tokens = this.tokenize(corpus);
            for (let i = 0; i < tokens.length - 1; i++) {
                const currentId = tokens[i];
                const nextId = tokens[i + 1];
                if (!this.bigramCounts.has(currentId)) {
                    this.bigramCounts.set(currentId, new Map());
                }
                const nextCounts = this.bigramCounts.get(currentId);
                nextCounts.set(nextId, (nextCounts.get(nextId) || 0) + 1);
            }
        }

        predictNextToken(currentTokenId) {
            const nextCounts = this.bigramCounts.get(currentTokenId);
            if (!nextCounts || nextCounts.size === 0) return null;

            let total = 0;
            for (let count of nextCounts.values()) total += count;

            const choice = Math.floor(Math.random() * total) + 1;
            let cumulative = 0;
            for (let [tokenId, count] of nextCounts.entries()) {
                cumulative += count;
                if (choice <= cumulative) return tokenId;
            }
            return null;
        }

        postTraining(corpus) {
            this.train(corpus);
            this.postTrainingHistory.push(corpus.slice(0, 80));
            return `post-training completed with ${this.tokenToId.size} tokens`;
        }

        instructionTuning(examples) {
            this.instructionTuningExamples.push(...examples);
            return `instruction tuning completed with ${examples.length} examples`;
        }

        describeArchitecture() {
            return JSON.stringify({
                embedding: `${this.embeddings.size} tokens with dimension ${this.embeddingDim}`,
                transformer: `${this.numLayers} layers and ${this.numHeads} heads`,
                self_attention: "enabled",
                deep_neural_network_layers: this.layerNames,
                parameters: this.parameters
            });
        }

        respond(prompt, maxLength = 20) {
            const promptIds = this.tokenize(prompt);
            if (!promptIds.length) return "";

            this.deepNeuralNetworkLayers(promptIds);

            const responseIds = [];
            let currentId = promptIds[promptIds.length - 1];
            for (let i = 0; i < maxLength; i++) {
                const nextId = this.predictNextToken(currentId);
                if (nextId === null) break;
                responseIds.push(nextId);
                currentId = nextId;
            }
            return this.detokenize(responseIds);
        }
    }

    // ==========================================
    // 🚀 OreAI Extension Main Class
    // ==========================================
    class OreAI {
        constructor() {
            // Global Settings
            this.apiKey = '';
            this.provider = 'Google Gemini';
            this.model = 'gemini-3.5-flash';
            this.temperature = 0.7;

            // Multi-Chatbot System
            this.chatbots = new Map();
            this.currentChatbotId = 'Default';
            this._createChatbot('Default', 'You are a helpful AI assistant.');

            // Sux Engine Setup (Legacy / Made by me)
            this.suxModel = new Sux();
            this.suxModel.train("hello how are you I am a simple ai model that learns from example text i can predict the next words and return a response based on a prompt");

            // Tracking Systems
            this.maxMessagesPerWindow = 0; // 0 = unlimited
            this.messageWindowMs = 2 * 60 * 60 * 1000; // Exactly 2 Hours
            this.messageTimestamps = [];
            this.totalMessagesSent = 0;
            this.responseDelayMs = 0;

            // v2.31 New Features
            this.currentResponseText = '';
            this.roleplayStrictness = 1.0; // 0.0 = relaxed persona, 2.0 = ultra strict roleplay

            // Utilities State
            this.lastGeneratedImageUrl = '';
            this.lastError = '';
            this.lastLatency = 0;
            this.status = 'Idle';
        }

        _createChatbot(name, systemPrompt) {
            this.chatbots.set(name, {
                systemPrompt: systemPrompt || 'You are a helpful assistant.',
                chatHistory: [],
                memoryLimit: 10,
                isPreset: false
            });
        }

        _getCurrentBot() {
            if (!this.chatbots.has(this.currentChatbotId)) {
                this._createChatbot(this.currentChatbotId, 'You are a helpful assistant.');
            }
            return this.chatbots.get(this.currentChatbotId);
        }

        _cleanMessageWindow() {
            const now = Date.now();
            this.messageTimestamps = this.messageTimestamps.filter(t => (now - t) < this.messageWindowMs);
        }

        _checkRateLimit() {
            this._cleanMessageWindow();
            if (this.maxMessagesPerWindow > 0 && this.messageTimestamps.length >= this.maxMessagesPerWindow) {
                const oldestInWindow = this.messageTimestamps[0];
                const waitTimeSec = Math.ceil((this.messageWindowMs - (Date.now() - oldestInWindow)) / 1000);
                throw new Error(`Rate limit reached (${this.maxMessagesPerWindow} msgs/2hrs). Try again in ${waitTimeSec} seconds.`);
            }
        }

        _buildSystemPrompt(bot) {
            let basePrompt = bot.systemPrompt;
            if (this.roleplayStrictness > 1.2) {
                basePrompt += ` [CRITICAL ROLEPLAY RULE: Stay strictly in character at all costs. Never break character or refer to yourself as an AI.]`;
            } else if (this.roleplayStrictness < 0.5) {
                basePrompt += ` [ROLEPLAY RULE: You may break character slightly if requested or if necessary to assist the user.]`;
            }
            return basePrompt;
        }

        getInfo() {
            return {
                id: 'oreAI',
                name: 'OreAI v2.31',
                color1: '#7b2cbf',
                color2: '#5a189a',
                color3: '#3c096c',
                blocks: [
                    // --- ⚙️ CONFIGURATION MANAGEMENT ---
                    '---',
                    '⚙️ Configuration & Models',
                    {
                        opcode: 'setApiKey',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set API key to [KEY]',
                        arguments: { KEY: { type: Scratch.ArgumentType.STRING, defaultValue: 'sk-...' } }
                    },
                    {
                        opcode: 'setProvider',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set AI provider to [PROVIDER]',
                        arguments: { PROVIDER: { type: Scratch.ArgumentType.STRING, menu: 'providers', defaultValue: 'Google Gemini' } }
                    },
                    {
                        opcode: 'setModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set model to [MODEL]',
                        arguments: { MODEL: { type: Scratch.ArgumentType.STRING, menu: 'models', defaultValue: 'gemini-3.5-flash' } }
                    },
                    {
                        opcode: 'setCustomModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set custom model ID to [MODEL_ID]',
                        arguments: { MODEL_ID: { type: Scratch.ArgumentType.STRING, defaultValue: 'deepseek-v4-pro' } }
                    },
                    {
                        opcode: 'getCurrentModel',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current model ID'
                    },
                    {
                        opcode: 'setTemperature',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set creativity (temperature) to [TEMP]',
                        arguments: { TEMP: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.7 } }
                    },
                    {
                        opcode: 'setResponseDelay',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set artificial response delay to [SECONDS] seconds',
                        arguments: { SECONDS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1.0 } }
                    },

                    // --- 🤖 MULTI-CHATBOT MANAGEMENT ---
                    '---',
                    '🤖 Multi-Chatbot Management',
                    {
                        opcode: 'createChatbot',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'create chatbot [NAME] with system prompt [PROMPT]',
                        arguments: {
                            NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'Shopkeeper' },
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'You are a merchant selling potions.' }
                        }
                    },
                    {
                        opcode: 'selectChatbot',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'select chatbot [NAME]',
                        arguments: { NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'Shopkeeper' } }
                    },
                    {
                        opcode: 'deleteChatbot',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'delete chatbot [NAME]',
                        arguments: { NAME: { type: Scratch.ArgumentType.STRING, defaultValue: 'Shopkeeper' } }
                    },
                    {
                        opcode: 'getCurrentChatbotName',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current chatbot name'
                    },
                    {
                        opcode: 'getChatbotList',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'all active chatbot names'
                    },

                    // --- 🎭 PERSONALITY & ROLEPLAY MANAGEMENT ---
                    '---',
                    '🎭 Personality & Roleplay',
                    {
                        opcode: 'setSystemPrompt',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set system prompt for current chatbot to [PROMPT]',
                        arguments: { PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'You are a helpful assistant.' } }
                    },
                    {
                        opcode: 'setPersonalityPreset',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'apply personality preset [PRESET] to current chatbot',
                        arguments: { PRESET: { type: Scratch.ArgumentType.STRING, menu: 'personalityPresets', defaultValue: 'Friendly NPC' } }
                    },
                    {
                        opcode: 'setRoleplayParameter',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set roleplay strictness parameter to [STRICTNESS]',
                        arguments: { STRICTNESS: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1.0 } }
                    },
                    {
                        opcode: 'isUsingPersonalityPreset',
                        blockType: Scratch.BlockType.BOOLEAN,
                        text: 'is current chatbot using personality preset?'
                    },
                    {
                        opcode: 'getSystemPrompt',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current chatbot system prompt'
                    },

                    // --- 💬 CHAT & MEMORY INSPECTION ---
                    '---',
                    '💬 Chat & Memory',
                    {
                        opcode: 'generateResponse',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'prompt AI [PROMPT]',
                        arguments: { PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'Hello! Tell me a fun fact.' } }
                    },
                    {
                        opcode: 'sendChatMessage',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'chat with selected bot memory [PROMPT]',
                        arguments: { PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'What items do you sell?' } }
                    },
                    {
                        opcode: 'getCurrentResponseText',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'ai current response text'
                    },
                    {
                        opcode: 'getMemoryStorage',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current chatbot memory storage (JSON)'
                    },
                    {
                        opcode: 'getMemoryCount',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'current chatbot stored message count'
                    },
                    {
                        opcode: 'clearChatHistory',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'clear current chatbot memory'
                    },

                    // --- 🏢 CORPORATE MANAGEMENT ---
                    '---',
                    '🏢 Corporate Management',
                    {
                        opcode: 'setMaxMessagesPerWindow',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set max messages allowed per 2 hours to [MAX]',
                        arguments: { MAX: { type: Scratch.ArgumentType.NUMBER, defaultValue: 20 } }
                    },
                    {
                        opcode: 'getMessagesSentCount',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'total messages sent'
                    },
                    {
                        opcode: 'getMessagesIn2HourWindow',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'messages sent in current 2-hour window'
                    },

                    // --- 🕹️ SUX MODEL (LEGACY / MADE BY ME) ---
                    '---',
                    '🕹️ Sux Model (Legacy / Made by me)',
                    {
                        opcode: 'suxTrain',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'Sux (legacy/Made by me) train on corpus [TEXT]',
                        arguments: { TEXT: { type: Scratch.ArgumentType.STRING, defaultValue: 'apple banana orange apple grape' } }
                    },
                    {
                        opcode: 'suxRespond',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'Sux (legacy/Made by me) generate from prompt [PROMPT] max tokens [MAX]',
                        arguments: {
                            PROMPT: { type: Scratch.ArgumentType.STRING, defaultValue: 'apple' },
                            MAX: { type: Scratch.ArgumentType.NUMBER, defaultValue: 10 }
                        }
                    },
                    {
                        opcode: 'suxDescribe',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'Sux (legacy/Made by me) architecture description'
                    },

                    // --- 🛠️ UTILITIES & DEBUGGING ---
                    '---',
                    '🛠️ Utilities',
                    {
                        opcode: 'getLastError',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last error'
                    },
                    {
                        opcode: 'getLatency',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last latency (ms)'
                    },
                    {
                        opcode: 'getStatus',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'AI status'
                    }
                ],
                menus: {
                    providers: {
                        acceptReporters: true,
                        items: ['Google Gemini', 'OpenAI', 'OpenRouter (Llama/DeepSeek/Mistral)']
                    },
                    models: {
                        acceptReporters: true,
                        items: [
                            'gemini-3.5-flash',
                            'gemini-3.1-flash-lite',
                            'gpt-5.4-mini',
                            'gpt-5.4',
                            'meta-llama/llama-4-maverick',
                            'meta-llama/llama-4-scout',
                            'meta-llama/llama-3.3-70b-instruct',
                            'deepseek/deepseek-v4-pro',
                            'deepseek/deepseek-v4-flash',
                            'deepseek/deepseek-r1',
                            'mistralai/mistral-small-4',
                            'mistralai/mistral-large-3'
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

        // --- ⚙️ CONFIG & MULTI-BOT IMPLEMENTATION ---
        setApiKey(args) { this.apiKey = Scratch.Cast.toString(args.KEY).trim(); }
        setProvider(args) { this.provider = Scratch.Cast.toString(args.PROVIDER); }
        setModel(args) { this.model = Scratch.Cast.toString(args.MODEL); }
        setCustomModel(args) { this.model = Scratch.Cast.toString(args.MODEL_ID).trim(); }
        getCurrentModel() { return this.model; }
        setTemperature(args) { this.temperature = Math.max(0, Math.min(2, Scratch.Cast.toNumber(args.TEMP))); }
        setResponseDelay(args) { this.responseDelayMs = Math.max(0, Scratch.Cast.toNumber(args.SECONDS) * 1000); }

        createChatbot(args) {
            const name = Scratch.Cast.toString(args.NAME).trim();
            const prompt = Scratch.Cast.toString(args.PROMPT);
            if (name) {
                this._createChatbot(name, prompt);
                this.currentChatbotId = name;
            }
        }

        selectChatbot(args) {
            const name = Scratch.Cast.toString(args.NAME).trim();
            if (this.chatbots.has(name)) {
                this.currentChatbotId = name;
            } else {
                this.lastError = `Chatbot "${name}" does not exist.`;
            }
        }

        deleteChatbot(args) {
            const name = Scratch.Cast.toString(args.NAME).trim();
            if (this.chatbots.has(name)) {
                this.chatbots.delete(name);
                if (this.currentChatbotId === name) {
                    const remainingKeys = Array.from(this.chatbots.keys());
                    if (remainingKeys.length > 0) {
                        this.currentChatbotId = remainingKeys[0];
                    } else {
                        this._createChatbot('Default', 'You are a helpful AI assistant.');
                        this.currentChatbotId = 'Default';
                    }
                }
            }
        }

        getCurrentChatbotName() { return this.currentChatbotId; }
        getChatbotList() { return Array.from(this.chatbots.keys()).join(', '); }

        // --- 🎭 PERSONALITY IMPLEMENTATION ---
        setSystemPrompt(args) {
            const bot = this._getCurrentBot();
            bot.systemPrompt = Scratch.Cast.toString(args.PROMPT);
            bot.isPreset = false;
        }

        setPersonalityPreset(args) {
            const bot = this._getCurrentBot();
            const preset = Scratch.Cast.toString(args.PRESET);
            switch (preset) {
                case 'Friendly NPC':
                    bot.systemPrompt = 'You are a cheerful NPC in a fantasy RPG.';
                    break;
                case 'Grumpy Shopkeeper':
                    bot.systemPrompt = 'You are a grumpy merchant who complains about customers and prices.';
                    break;
                case 'Wise Mentor':
                    bot.systemPrompt = 'You are an ancient wizard giving mystical guidance.';
                    break;
                case 'Sci-Fi AI Assistant':
                    bot.systemPrompt = 'You are a spaceship AI speaking with concise tactical terminology.';
                    break;
                case 'Concise Assistant (Max 15 words)':
                    bot.systemPrompt = 'Helpful assistant. Rule: Never reply with more than 15 words.';
                    break;
            }
            bot.isPreset = true;
        }

        setRoleplayParameter(args) {
            this.roleplayStrictness = Math.max(0, Math.min(2, Scratch.Cast.toNumber(args.STRICTNESS)));
        }

        isUsingPersonalityPreset() { return this._getCurrentBot().isPreset; }
        getSystemPrompt() { return this._getCurrentBot().systemPrompt; }

        // --- 💬 MEMORY IMPLEMENTATION ---
        getCurrentResponseText() { return this.currentResponseText; }
        getMemoryStorage() { return JSON.stringify(this._getCurrentBot().chatHistory); }
        getMemoryCount() { return this._getCurrentBot().chatHistory.length; }
        clearChatHistory() { this._getCurrentBot().chatHistory = []; }

        // --- 🏢 CORPORATE MANAGEMENT IMPLEMENTATION ---
        setMaxMessagesPerWindow(args) { this.maxMessagesPerWindow = Math.max(0, Scratch.Cast.toNumber(args.MAX)); }
        getMessagesSentCount() { return this.totalMessagesSent; }
        getMessagesIn2HourWindow() {
            this._cleanMessageWindow();
            return this.messageTimestamps.length;
        }

        // --- 🕹️ SUX (LEGACY / MADE BY ME) IMPLEMENTATION ---
        suxTrain(args) {
            const corpus = Scratch.Cast.toString(args.TEXT);
            this.suxModel.train(corpus);
        }

        suxRespond(args) {
            const prompt = Scratch.Cast.toString(args.PROMPT);
            const max = Scratch.Cast.toNumber(args.MAX);
            const text = this.suxModel.respond(prompt, max);
            this.currentResponseText = text;
            return text;
        }

        suxDescribe() { return this.suxModel.describeArchitecture(); }

        // --- 💬 CHATBOT INTERACTION EXECUTOR ---
        async generateResponse(args) { return await this._executeRequest(Scratch.Cast.toString(args.PROMPT), false); }
        async sendChatMessage(args) { return await this._executeRequest(Scratch.Cast.toString(args.PROMPT), true); }

        getLastError() { return this.lastError; }
        getLatency() { return this.lastLatency; }
        getStatus() { return this.status; }

        async _executeRequest(userPrompt, useMemory) {
            if (!this.apiKey) {
                this.lastError = 'API Key is missing.';
                return 'Error: API Key is missing.';
            }

            try {
                this._checkRateLimit();
            } catch (limitErr) {
                this.lastError = limitErr.message;
                return `Error: ${this.lastError}`;
            }

            this.status = 'Thinking...';
            const startTime = Date.now();
            const bot = this._getCurrentBot();

            try {
                if (this.responseDelayMs > 0) {
                    await new Promise(resolve => setTimeout(resolve, this.responseDelayMs));
                }

                let responseText = '';
                if (this.provider === 'Google Gemini') {
                    responseText = await this._callGemini(userPrompt, useMemory, bot);
                } else if (this.provider.includes('OpenRouter')) {
                    responseText = await this._callOpenRouter(userPrompt, useMemory, bot);
                } else {
                    responseText = await this._callOpenAI(userPrompt, useMemory, bot);
                }

                // Store in response state reporter
                this.currentResponseText = responseText;

                if (useMemory) {
                    bot.chatHistory.push({ role: 'user', content: userPrompt });
                    bot.chatHistory.push({ role: 'assistant', content: responseText });
                    while (bot.chatHistory.length > bot.memoryLimit * 2) {
                        bot.chatHistory.shift();
                    }
                }

                // Log message for 2-hour window and total stats
                this.messageTimestamps.push(Date.now());
                this.totalMessagesSent++;

                this.lastLatency = Date.now() - startTime;
                this.status = 'Idle';
                return responseText;
            } catch (err) {
                this.lastError = err.message || String(err);
                this.status = 'Error';
                return `Error: ${this.lastError}`;
            }
        }

        async _callOpenAI(prompt, useMemory, bot) {
            const url = 'https://api.openai.com/v1/chat/completions';
            const sysPrompt = this._buildSystemPrompt(bot);
            let messages = [{ role: 'system', content: sysPrompt }];
            if (useMemory) messages = messages.concat(bot.chatHistory);
            messages.push({ role: 'user', content: prompt });

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}` },
                body: JSON.stringify({ model: this.model, temperature: this.temperature, messages })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
            return data.choices?.[0]?.message?.content || '';
        }

        async _callOpenRouter(prompt, useMemory, bot) {
            const url = 'https://openrouter.ai/api/v1/chat/completions';
            const sysPrompt = this._buildSystemPrompt(bot);
            let messages = [{ role: 'system', content: sysPrompt }];
            if (useMemory) messages = messages.concat(bot.chatHistory);
            messages.push({ role: 'user', content: prompt });

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`,
                    'HTTP-Referer': 'https://turbowarp.org',
                    'X-Title': 'OreAI Scratch Extension'
                },
                body: JSON.stringify({ model: this.model, temperature: this.temperature, messages })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
            return data.choices?.[0]?.message?.content || '';
        }

        async _callGemini(prompt, useMemory, bot) {
            const endpointModel = this.model.includes('gemini') ? this.model : 'gemini-3.5-flash';
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${endpointModel}:generateContent?key=${this.apiKey}`;
            const sysPrompt = this._buildSystemPrompt(bot);

            let contents = [];
            if (useMemory) {
                contents = bot.chatHistory.map(item => ({
                    role: item.role === 'assistant' ? 'model' : 'user',
                    parts: [{ text: item.content }]
                }));
            }
            contents.push({ role: 'user', parts: [{ text: prompt }] });

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    systemInstruction: { parts: [{ text: sysPrompt }] },
                    generationConfig: { temperature: this.temperature },
                    contents
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
            return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
        }
    }

    Scratch.extensions.register(new OreAI());
})(Scratch);
