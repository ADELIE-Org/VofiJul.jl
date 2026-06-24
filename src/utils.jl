pad_to_ndim(vec, ::Type{T} = vofi_real) where {T} = begin
    out = zero(MVector{NDIM, T})
    for i in 1:min(length(vec), NDIM)
        out[i] = T(vec[i])
    end
    out
end

@inline function axis_index(dir)
    for i in 1:length(dir)
        if dir[i] != 0
            return i
        end
    end
    throw(ArgumentError("direction vector must have a non-zero component"))
end

@inline axis_length(dir, hvec) = hvec[axis_index(dir)]
