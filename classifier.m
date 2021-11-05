%r = audiorecorder(44100,16,1);
%record(r);
%stop(r);
%play(r);
%mySpeech = getaudiodata(r);

Fs = 44100;
N = round(Fs*0.023); % 1014
L = round(Fs*0.011); % 485
filename = 'audio1.wav';
%audiowrite(filename,mySpeech,Fs,'BitsPerSample',16);

%Read the data back into MATLAB using audioread.
[mySpeech,Fs] = audioread(filename);

%sound(mySpeech,Fs);
%plot(mySpeechR);hold;stem(mySpeechR);figure(1)



%
% Applying BandPass Filter
%
mySpeechF = bandpass(mySpeech,[60 5000],44100);
t = linspace(0,1,length(mySpeech));

figure(1);
plot(t,mySpeech,t,mySpeechF)
xlabel('Time (s)')
ylabel('Amplitude')
legend('Original Signal','Filtered Data')


%
% Resampling signal
%
[P,Q] = rat(22050/44100);
mySpeechR = resample(mySpeechF,P,Q);



%
%SEGMENTING SIGNAL
%
% determine the number of rows/frames
m = length(mySpeechR);
numFrames = floor((m-1)/N+1);

% allocate memory to the frame matrix
frameData = zeros(numFrames,N);

% Now, extract the frames from the input data
for k=1:numFrames
   startAtIdx = (k-1)*N+1;
   if k~=numFrames
      frameData(k,:) = mySpeechR(startAtIdx:startAtIdx+N-1);
   else
      % handle this case separately in case the number of input samples
      % does not divide evenly by the window size
      frameData(k,1:m-startAtIdx+1) = mySpeechR(startAtIdx:end);
   end
end



% Calculating Energy envelope
E = short_term_energy(mySpeechR,N,1);
figure(2)
plot(E);

hold
figure(3)
stem(E);

C = 85;
figure(4)
plot(C+10*log10(E));




% Calculating Zero-Crosssing-Rate for each frame
Zcr = ShortTimeZeroCrossingRate(mySpeechR,N,N);
if mod(length(mySpeechR), N)~= 0
    Zcr(length(Zcr) + 1) = 0.1;
end



%
% Classifying foreground and background segments
%
n = size(frameData,1);   %number of Rows
per = length(E)/length(mySpeechR);
currentLimit = 0;
limit = 1;
pointer = 1;
AvgE = zeros(1,n);
ClassArray = zeros(1,n);
%foreach frame
for i=1:n
    avg = 0;
    frame = reshape(frameData(i,:),[length(frameData(i,:)),1]);
    
    limit = i*per;  % 1011.09665
    Evalues = round(per*N);  % 1011
    currentLimit = currentLimit + Evalues;
    if (limit-currentLimit >= 1)
        Evalues = Evalues + 1;  % 1012
        currentLimit = currentLimit + 1;
        
        if (pointer + Evalues > length(E))
            Evalues = length(E)- pointer;
        end
    else
        if (pointer + Evalues > length(E))
            Evalues = length(E)- pointer;
        end
    end
    
    for j = pointer:pointer + Evalues
        avg = avg + E(j);
    end
    avg = avg/Evalues;
    
    AvgE(1,i) = avg;
    pointer = pointer + Evalues;
    
    %
    % Classifying foreground and background segments
    %
    if (C+10*log10(AvgE(1,i)) > 51 && Zcr(1,i) < 0.09)
        ClassArray(1,i) = 1;
    end
end


figure(5);
plot(ClassArray);


%
%CREATING ARRAY OF CLASS 1 FRAMES
%
columns = 0;
for i=1:n
    if (ClassArray(1,i) == 1)
        columns = columns + 1;
    end
end

position = zeros(1,columns);
index = 1;
for i=1:n
    if (ClassArray(1,i) == 1)
        position(1,index) = i;
        index = index + 1;
    end
end

SpeechDataArray = zeros(N,columns);
for j=1:columns
    SpeechDataArray(:,j) = frameData(position(1,j),:);
end



%
% Checking if sentence contains more than five and less than ten digits
%
% finding number of digits in signal
k = 1;
numberOfSignals = 0;
for j=1:length(position)
    if (position(j) - 1 ~= position(k))
        numberOfSignals = numberOfSignals + 1;
    end
    if (j ~= 1)
        k = k + 1;
    end
end


%
% Mel-Spectrogram
%
if (length(numberOfSignals) < 5 && length(numberOfSignals) > 10)
    disp('Invalid audio input! Sentence contains less than 5 or more than 10 digits. Try again with another signal as input.');
else
    [S,F,T] = melSpectrogram(SpeechDataArray,Fs,'Window',hann(5,'periodic'),'OverlapLength',4,'NumBands',64,'FFTLength',1014);
    figure(6);
    imagesc(10*log10(S(:,:,1).^2));
end










