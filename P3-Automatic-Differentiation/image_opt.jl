#
using ImageView
using TestImages
using Images

function conv_1!(result, input, filter)
    # input and result: 512 x 512
    # filter is n x n, n = 2*[1, 100] + 1 ??
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)
    result_r, result_c = size(result)

    filter_middle = convert(Int32, (filter_r + 1)//2)
    m = convert(Int32, (filter_r - 1)//2)

    if filter_r != filter_c
        throw(DomainError(filter, "Filter row and column must be equals"))
    end

    # Because the filter is 3x3, then the conv matrix will be reduced by 2.
    # result = zeros(input_r-2, input_c-2)

    for i in 1:result_r
        for j in 1:result_c
            for fi in -m:m
                for fj in -m:m
                    Ii = mod(i + fi - 1,input_r) + 1
                    Ij = mod(j + fj - 1,input_c) + 1
                    result[i,j] += input[Ii,Ij]*filter[filter_middle+fi,filter_middle+fj]
                    # extend for an arbitrary filter size
                end
            end
        end
    end

    result = min.(result, 1.0)
    result = max.(result, 0.0)
    return result
end

img = testimage("mandrill")
img_gray= Gray.(img)
# retrieve in 0-255 format
img_array = convert(Array{Float64,2},img_gray)
n=21
filter = rand(n,n)/n/n #3.3e-1*[1 1 1; 0 1 1; 1 0 0]
img_result = similar(img_array)
conv_1!(img_result,img_array, filter)
imshow(img_result)

# objective function = (relative distance of images) + sum.(abs.(filter))
# Newton (objective_function(Jü_orig, Jü_TetGen, filter))