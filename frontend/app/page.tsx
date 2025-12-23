"use client";

import React, { useState, useRef } from 'react';
import {
  Volume2,
  Play,
  Pause,
  Download,
  Loader2,
  Settings,
  BookOpen,
} from 'lucide-react';

type VoiceOption = {
  id: string;
  name: string;
  lang: string;
};

export default function TTSService() {
  const [text, setText] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [voice, setVoice] = useState<string>('en-US-Standard-A');
  const [speed, setSpeed] = useState<number>(1.0);
  const [pitch, setPitch] = useState<number>(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const TTS_API_URL = "http://localhost:8000/tts_wav";

  const [maxChars, setMaxChars] = useState<number>(200);
  const [crossFadeDuration, setCrossFadeDuration] = useState<number>(0.15);

  // const voices: VoiceOption[] = [
  //   { id: 'en-US-Standard-A', name: 'US English - Female 1', lang: 'en-US' },
  //   { id: 'en-US-Standard-B', name: 'US English - Male 1', lang: 'en-US' },
  //   { id: 'en-US-Standard-C', name: 'US English - Female 2', lang: 'en-US' },
  //   { id: 'en-US-Standard-D', name: 'US English - Male 2', lang: 'en-US' },
  //   { id: 'en-GB-Standard-A', name: 'UK English - Female', lang: 'en-GB' },
  //   { id: 'en-GB-Standard-B', name: 'UK English - Male', lang: 'en-GB' },
  // ];

  const handleSynthesize = async (): Promise<void> => {
    if (!text.trim()) return;

    setIsLoading(true);
    setIsPlaying(false);

    try {
      // cleanup previous audio
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const response = await fetch(TTS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          speed,
          max_chars: 200,
          cross_fade_duration: 0.15,
        }),
      });

      if (!response.ok) {
        throw new Error(`TTS failed: ${response.status}`);
      }

      // backend returns WAV binary
      const audioBlob = await response.blob();

      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

    } catch (error) {
      console.error("TTS Error:", error);
      alert("Failed to synthesize speech.");
    } finally {
      setIsLoading(false);
    }
  };

  const togglePlayPause = (): void => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }

    setIsPlaying((prev) => !prev);
  };

  const handleDownload = (): void => {
    if (!audioUrl) return;

    const a = document.createElement('a');
    a.href = audioUrl;
    a.download = 'speech.mp3';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const loadSampleText = (sample: string): void => {
    setText(sample);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-8 text-white">
            <div className="flex items-center gap-3 mb-2">
              <Volume2 className="w-10 h-10" />
              <h1 className="text-4xl font-bold">Text-to-Speech</h1>
            </div>
            <p className="text-indigo-100">
              Convert your text into natural-sounding speech
            </p>
          </div>

          <div className="p-8">
            {/* Sample Texts */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen className="w-5 h-5 text-indigo-600" />
                {/* <label className="text-sm font-semibold text-gray-700">
                  Quick Samples:
                </label> */}
              </div>
              {/* <div className="flex flex-wrap gap-2">
                {sampleTexts.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => loadSampleText(sample)}
                    className="px-4 py-2 text-sm bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-lg transition-colors"
                  >
                    Sample {idx + 1}
                  </button>
                ))}
              </div> */}
            </div>

            {/* Text Input */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Enter Text
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type or paste the text you want to convert to speech..."
                className="w-full h-40 px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:outline-none resize-none transition-colors text-gray-800"
                maxLength={5000}
              />
              <div className="flex justify-between items-center mt-2">
                <span className="text-sm text-gray-500">
                  {text.length} / 5000 characters
                </span>
                {text.length > 0 && (
                  <button
                    onClick={() => setText('')}
                    className="text-sm text-red-500 hover:text-red-600"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Settings Toggle */}
            <button
              onClick={() => setShowSettings((prev) => !prev)}
              className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 mb-4 font-medium"
            >
              <Settings className="w-5 h-5" />
              {showSettings ? 'Hide' : 'Show'} Voice Settings
            </button>

            {/* Voice Settings */}
            {showSettings && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 p-4 bg-gray-50 rounded-xl">
                
                {/* Speed */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Speed: {speed.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    value={speed}
                    onChange={(e) => setSpeed(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Max Characters */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Characters: {maxChars}
                  </label>
                  <input
                    type="number"
                    min={50}
                    max={5000}
                    step={50}
                    value={maxChars}
                    onChange={(e) => setMaxChars(parseInt(e.target.value, 10))}
                    className="w-full px-2 py-1 border border-gray-300 rounded-lg focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                {/* Cross-fade Duration */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cross-fade Duration: {crossFadeDuration}s
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={crossFadeDuration}
                    onChange={(e) => setCrossFadeDuration(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3 mb-6">
              <button
                onClick={handleSynthesize}
                disabled={!text.trim() || isLoading}
                className="flex-1 min-w-[200px] flex items-center justify-center gap-2 px-6 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Synthesizing...
                  </>
                ) : (
                  <>
                    <Volume2 className="w-5 h-5" />
                    Generate Speech
                  </>
                )}
              </button>

              {audioUrl && (
                <>
                  <button
                    onClick={togglePlayPause}
                    className="flex items-center justify-center gap-2 px-6 py-4 bg-green-600 text-white rounded-xl font-semibold hover:bg-green-700 transition-colors shadow-lg hover:shadow-xl"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-5 h-5" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Play
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleDownload}
                    className="flex items-center justify-center gap-2 px-6 py-4 bg-gray-600 text-white rounded-xl font-semibold hover:bg-gray-700 transition-colors shadow-lg hover:shadow-xl"
                  >
                    <Download className="w-5 h-5" />
                    Download
                  </button>
                </>
              )}
            </div>

            {/* Audio Element */}
            {audioUrl && (
              <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                className="hidden"
              />
            )}

            {/* Info Card */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <h3 className="font-semibold text-blue-900 mb-2">
                Backend Integration Required
              </h3>
              <p className="text-sm text-blue-800">
                This UI requires backend with a TTS service.
                Configure your backend endpoint at{' '}
                <code className="bg-blue-100 px-2 py-1 rounded">
                  /tts_wav
                </code>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
