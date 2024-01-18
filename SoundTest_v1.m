%%% Sound test v1%%%
%%% - check sound and speaker configuration
%%%   play sin tone as many as the channel number and pink noise

%%% Evaluation - Peak of ERP for t-test%% 
%%% - evoked responce
%%%
%%% required Add-ons
%%% - 
%%% required functions
%%% - 
%%% required setting files
%%% - 

%%% v1  
%%% 20231114 for multichannel sound test

function [] = SoundTest_v1(numSpk, numIter) 

% numSpk  = 2; %number of speakers
% numIter = 2; %numbert of iteration 

%% audio info
out = PsychPortAudio('GetDevices'); %get sound device information for Psychtool box
prompt = 'Choose Audio device'; % prompt message
[DevID_indx,tf] = listdlg('PromptString',prompt,'SelectionMode','single','ListSize',[200 150],'ListString',{out.DeviceName}); % option selection window
DevID = DevID_indx - 1; %the ID of Psychtoolbox starts with 0
buffer = 100; %playback buffer size
fs = 48000; % sample rate for audio
volume = 1; % stimuli volume
vol_sin = 0.3; %sintone volume

dur_pink = 6.5; %pink noise duration

InitializePsychSound %PsychTool Box
pahandle = PsychPortAudio('Open', DevID, [], 2, fs, numSpk, buffer); % must be the same parameters with the current AUDIO Hardware such as FS, etc.

%% Sin wave Generation

dt = 1/fs;           % seconds per sample
SinDur = .05;             % duration [sec]
t = (0:dt:SinDur-dt)';   % seconds
F = 1000;                % Sine wave frequency (hertz)
sinwave = sin(2*pi*F*t)*vol_sin;
    
%% Pink Noise Generation

Pnoise=pinknoise(fs*dur_pink);

%% play

for i=1:numIter
    for j = 1:numSpk
        for k = 1:j
            sintone = zeros(numSpk,length(sinwave));
            sintone(j,:) = sinwave;
            PsychPortAudio('Volume', pahandle, volume);
            PsychPortAudio('FillBuffer', pahandle, sintone);   % load stimuli stimulus
            PsychPortAudio('Start',pahandle);               % playback
            WaitSecs(0.1);
        end
    
        testsound = zeros(numSpk,length(Pnoise));
        testsound(j,:) = Pnoise;
        PsychPortAudio('Volume', pahandle, volume);
        PsychPortAudio('FillBuffer', pahandle, testsound);   % load stimuli stimulus
        PsychPortAudio('Start',pahandle);               % playback
    end
end


end
 
 
        