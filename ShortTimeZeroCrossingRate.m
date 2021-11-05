% FUNCTION ShortTimeZeroCrossingRate : windowLength and step in # of samples

function ZCR = ShortTimeZeroCrossingRate(signal, windowLength,step)

curPos = 1;
L = length(signal);
E = [];
frameCounter=1;
while (curPos+windowLength-1<=L)    
    window = (signal(curPos:curPos+windowLength-1));
    temp=0;
    for i=2:windowLength
        if window(i)>=0 && window(i-1)<0
            temp=temp+2;
        end
        if window(i)<0 && window(i-1)>=0
            temp=temp+2;
        end
    end    
    ZCR(frameCounter) = (1/(2*windowLength))*temp;
    curPos = curPos + step;
    frameCounter=frameCounter+1;
end
