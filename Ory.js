(function (Scratch) {
    'use strict';

    if (!Scratch.extensions.unsandboxed) {
        throw new Error('OreAI must be run unsandboxed to perform network requests.');
    }

    class OreAI {
        constructor() {
            this.apiKey = '';
            this.provider = 'Google Gemini';
            this.model = 'gemini-3.5-flash';
            this.systemPrompt = 'You are a helpful AI assistant integrated into a Scratch project.';
            this.lastError = '';
        }

        getInfo() {
            return {
                id: 'oreAI',
                name: 'OreAI',
                color1: '#7b2cbf',
                color2: '#5a189a',
                color3: '#3c096c',
                blocks: [
                    {
                        opcode: 'setApiKey',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set API key to [KEY]',
                        arguments: {
                            KEY: {
                                type: Scratch.ArgumentType.STRING,
                                defaultValue: 'AIzaSy...'
                            }
                        }
                    },
                    {
                        opcode: 'setProvider',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set AI provider to [PROVIDER]',
                        arguments: {
                            PROVIDER: {
                                type: Scratch.ArgumentType.STRING,
                                menu: 'providers',
                                defaultValue: 'Google Gemini'
                            }
                        }
                    },
                    {
                        opcode: 'setModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set model to [MODEL]',
                        arguments: {
                            MODEL: {
                                type: Scratch.ArgumentType.STRING,
                                menu: 'models',
                                defaultValue: 'gemini-3.5-flash'
                            }
                        }
                    },
                    {
                        opcode: 'setCustomModel',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set custom model ID to [MODEL_ID]',
                        arguments: {
                            MODEL_ID: {
                                type: Scratch.ArgumentType.STRING,
                                defaultValue: 'gemini-3.5-flash'
                            }
                        }
                    },
                    {
                        opcode: 'setSystemPrompt',
                        blockType: Scratch.BlockType.COMMAND,
                        text: 'set system prompt to [PROMPT]',
                        arguments: {
                            PROMPT: {
                                type: Scratch.ArgumentType.STRING,
                                defaultValue: 'You are a helpful assistant.'
                            }
                        }
                    },
                    '---',
                    {
                        opcode: 'generateResponse',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'prompt AI [PROMPT]',
                        arguments: {
                            PROMPT: {
                                type: Scratch.ArgumentType.STRING,
                                defaultValue: 'Hello! Tell me a fun fact.'
                            }
                        }
                    },
                    {
                        opcode: 'getLastError',
                        blockType: Scratch.BlockType.REPORTER,
                        text: 'last error',
                        disableMonitor: true
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
                    }
                }
            };
        }

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

        setSystemPrompt(args) {
            this.systemPrompt = Scratch.Cast.toString(args.PROMPT);
        }

        getLastError() {
            return this.lastError;
        }

        async generateResponse(args) {
            const prompt = Scratch.Cast.toString(args.PROMPT);
            if (!this.apiKey) {
                this.lastError = 'API Key is missing.';
                return 'Error: API Key is missing.';
            }

            try {
                if (this.provider === 'Google Gemini') {
                    return await this._callGemini(prompt);
                } else {
                    return await this._callOpenAI(prompt);
                }
            } catch (err) {
                this.lastError = err.message || String(err);
                return `Error: ${this.lastError}`;
            }
        }

        async _callOpenAI(prompt) {
            const url = 'https://api.openai.com/v1/chat/completions';
            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.apiKey}`
                },
                body: JSON.stringify({
                    model: this.model,
                    messages: [
                        { role: 'system', content: this.systemPrompt },
                        { role: 'user', content: prompt }
                    ]
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error?.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            return data.choices?.[0]?.message?.content || '';
        }

        async _callGemini(prompt) {
            const endpointModel = this.model.includes('gemini') ? this.model : 'gemini-3.5-flash';
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${endpointModel}:generateContent?key=${this.apiKey}`;

            const response = await Scratch.fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    systemInstruction: {
                        parts: [{ text: this.systemPrompt }]
                    },
                    contents: [
                        {
                            role: 'user',
                            parts: [{ text: prompt }]
                        }
                    ]
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