function vofi_reorder!(baser, basei, nsub)
    for j in 3:nsub
        ls = baser[j]
        ks = basei[j]
        i = j - 1
        while i > 1 && baser[i] > ls
            baser[i + 1] = baser[i]
            basei[i + 1] = basei[i]
            i -= 1
        end
        baser[i + 1] = ls
        basei[i + 1] = ks
    end
end

function vofi_rm_segs!(baser, basei, nsub)
    eps2 = (2 * EPS_LOC, EPS_SEGM)
    i = 1
    bs = baser[1]
    be = baser[nsub + 1]
    while i <= nsub
        ds = baser[i + 1] - baser[i]
        ks = basei[i + 1] * basei[i]
        eps = eps2[ks == 0 ? 1 : 2]
        if ds < eps
            if basei[i] != 1 && basei[i + 1] == 1
                baser[i] = baser[i + 1]
            end
            for j in i+1:nsub
                baser[j] = baser[j + 1]
                basei[j] = basei[j + 1]
            end
            nsub -= 1
        else
            i += 1
        end
    end
    baser[1] = bs
    baser[nsub + 1] = be
    return nsub
end

function vofi_sector_new!(sign_sect, nsect, ndire, nsub, iside, isect, down2up)
    for i in 1:nsub
        if sign_sect[iside, 1] * sign_sect[isect, i] > 0
            nsect[i] = sign_sect[iside, 1] < 0 ? 1 : 0
            ndire[i] = 0
        else
            nsect[i] = -1
            ndire[i] = down2up
        end
    end
end

function vofi_sector_old!(ws::VofiWorkspace, impl_func, par, x0, h0, base, pdir, sdir, nsect, ndire, nsub)
    T = promote_type(eltype(x0), eltype(h0))
    x1 = ws.so_x1
    x2 = ws.so_x2
    fse = ws.so_fse
    for i in 1:nsub
        mdpt = 0.5 * (base[i + 1] + base[i])
        for j in 1:NDIM
            x1[j] = x0[j] + sdir[j] * mdpt
            x2[j] = x1[j] + pdir[j] * h0[j]
        end
        fse[1] = call_integrand(impl_func, par, x1)
        fse[2] = call_integrand(impl_func, par, x2)
        if fse[1] * fse[2] <= 0
            nsect[i] = -1
            ndire[i] = (fse[1] < 0 || fse[2] > 0) ? 1 : -1
        else
            nsect[i] = fse[1] < 0 ? 1 : 0
            ndire[i] = 0
        end
    end
end

function vofi_get_limits_2D(ws::VofiWorkspace, impl_func, par, x0, h0, f0, xfsp, base, pdir, sdir, nsect, ndire)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    basei = ws.gl2_basei; fill!(basei, 0)
    baser[1] = 0
    basei[1] = 1
    nsub = 1
    down2up = 1
    atleast1 = false
    iside = 1
    isect = 2
    sign_sect = ws.gl2_sign_sect; fill!(sign_sect, 0)
    hs = zero(T)
    for i in 1:NDIM
        hs += sdir[i] * h0[i]
    end
    nbt = ws.gl2_nbt; fill!(nbt, 0)

    fse = ws.gl2_fse
    x1 = ws.gl2_x1
    for k in 0:1
        fse[1] = f0[k + 1, 1]
        fse[2] = f0[k + 1, 2]
        fsum = fse[1] + fse[2]
        if xfsp.isc[k + 2] == 0
            atleast1 = true
            nbt[k + 1] = 1
            iside = k + 1
            isect = 2 - k
            if fsum > 0
                sign_sect[k + 1, 1] = 1
            elseif fsum < 0
                sign_sect[k + 1, 1] = -1
            end
            if (k == 0 && sign_sect[k + 1, 1] == 1) ||
               (k == 1 && sign_sect[k + 1, 1] == -1)
                down2up = -1
            end
        else
            for i in 1:NDIM
                x1[i] = x0[i] + k * pdir[i] * h0[i]
            end
            inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsp,
                                                 baser, sdir, hs, nsub, xfsp.isc[k + 2])
            nsub += inters
            for i in 1:inters
                basei[nsub - i + 1] = 1
            end
            nbt[k + 1] = inters + 1
            if inters == 1
                if fse[1] < 0
                    sign_sect[k + 1, 1] = -1
                    sign_sect[k + 1, 2] = 1
                else
                    sign_sect[k + 1, 1] = 1
                    sign_sect[k + 1, 2] = -1
                end
            else
                if fsum > 0
                    sign_sect[k + 1, 1] = 1
                    sign_sect[k + 1, 2] = -1
                    sign_sect[k + 1, 3] = 1
                else
                    sign_sect[k + 1, 1] = -1
                    sign_sect[k + 1, 2] = 1
                    sign_sect[k + 1, 3] = -1
                end
            end
        end
    end

    ncheck = atleast1 ? max(nbt...) : 0
    baser[nsub + 1] = hs
    basei[nsub + 1] = 1
    vofi_reorder!(baser, basei, nsub + 1)
    nsub = vofi_rm_segs!(baser, basei, nsub)

    if ncheck == nsub
        vofi_sector_new!(sign_sect, nsect, ndire, nsub, iside, isect, down2up)
    else
        vofi_sector_old!(ws, impl_func, par, x0, h0, baser, pdir, sdir, nsect, ndire, nsub)
    end

    return nsub
end

function vofi_get_limits_3D(ws::VofiWorkspace, impl_func, par, x0, h0, f0, xfsp, base, pdir, sdir, tdir)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    basei = ws.gl3_basei; fill!(basei, 0)
    baser[1] = 0
    basei[1] = 1
    nsub = 1
    hs = ht = zero(T)
    for i in 1:NDIM
        hs += sdir[i] * h0[i]
        ht += tdir[i] * h0[i]
    end
    xs = ws.gl3_xs
    xp = ws.gl3_xp
    xt = ws.gl3_xt
    fse = ws.gl3_fse
    xfsl = ws.gl3_xfsl

    for m in 0:1
        for i in 1:NDIM
            xp[i] = x0[i] + m * pdir[i] * h0[i]
        end
        for n in 0:1
            l0 = 2 * m + n + 1
            if xfsp[l0].isc[2] != 0
                for i in 1:NDIM
                    xs[i] = xp[i] + n * sdir[i] * h0[i]
                end
                fse[1] = f0[m + 1, n + 1, 1]
                fse[2] = f0[m + 1, n + 1, 2]
                inters = vofi_get_side_intersections(ws, impl_func, par, fse, xs,
                                                     xfsp[l0], baser, tdir, ht,
                                                     nsub, xfsp[l0].isc[2])
                nsub += inters
                for i in 1:inters
                    basei[nsub - i + 1] = 1
                end
                for i in 1:NDIM
                    xt[i] = xs[i] + baser[nsub - 1] * tdir[i]
                end
                consi = vofi_check_line_consistency(ws, impl_func, par, xt, sdir, hs, n, xfsl)
                if inters > 1 && consi == 0
                    for i in 1:NDIM
                        xt[i] = xs[i] + baser[nsub - 2] * tdir[i]
                    end
                    consi = vofi_check_line_consistency(ws, impl_func, par, xt, sdir, hs, n, xfsl)
                end
                if consi > 0
                    inters = vofi_get_ext_intersections(ws, impl_func, par, xp, h0, xfsl,
                                                        baser, sdir, tdir, nsub)
                    nsub += inters
                    for i in 1:inters
                        basei[nsub - i + 1] = 0
                    end
                end
            end
        end
        if xfsp[5].isc[m + 2] != 0
            inters = vofi_get_ext_intersections(ws, impl_func, par, xp, h0, xfsp[5],
                                                baser, sdir, tdir, nsub)
            nsub += inters
            for i in 1:inters
                basei[nsub - i + 1] = 0
            end
        end
    end

    baser[nsub + 1] = ht
    basei[nsub + 1] = 1
    vofi_reorder!(baser, basei, nsub + 1)
    nsub = vofi_rm_segs!(baser, basei, nsub)
    return nsub
end

function vofi_get_limits_4D(ws::VofiWorkspace, impl_func, par, x0, h0, f0, xfsp::XFSP4D, base, pdir, sdir, tdir, qdir)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    qlen = axis_length(qdir, h0)
    baser[1] = 0
    cuts = ws.gl4_cuts
    empty!(cuts)
    push!(cuts, zero(T))
    push!(cuts, qlen)
    for edge in xfsp.edges
        if edge.isc[1] == 1
            sval = clamp(edge.sval, zero(T), qlen)
            push!(cuts, sval)
        end
    end
    sort!(cuts)
    nsub = 0
    last = zero(T)
    for idx in 2:length(cuts)
        val = cuts[idx]
        if val - last <= EPS_LOC
            continue
        end
        if nsub == NSEG
            baser[nsub + 1] = qlen
            return nsub
        end
        nsub += 1
        baser[nsub + 1] = val
        last = val
    end
    if nsub == 0
        nsub = 1
        baser[2] = qlen
    elseif abs(baser[nsub + 1] - qlen) > EPS_LOC
        if nsub < NSEG
            nsub += 1
        end
        baser[nsub + 1] = qlen
    end
    return nsub
end

function vofi_check_plane(ws::VofiWorkspace, impl_func, par, x0, h0, xfs_pt, base, pdir, sdir, nsect, ndire)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    basei = ws.cp_basei; fill!(basei, 0)
    baser[1] = 0
    basei[1] = 1
    nsub = 1
    hs = zero(T)
    for i in 1:NDIM
        hs += sdir[i] * h0[i]
        xfs_pt.isc[i] = 0
    end
    x1 = ws.cp_x1
    x2 = ws.cp_x2
    fse = ws.cp_fse
    xfsl = ws.cp_xfsl; reset!(xfsl)
    nbt = ws.cp_nbt; fill!(nbt, 0)
    sign_sect = ws.cp_sign_sect; fill!(sign_sect, 0)
    atleast1 = false
    down2up = 1
    iside = 1
    isect = 2

    for k in 0:1
        for i in 1:NDIM
            x1[i] = x0[i] + k * pdir[i] * h0[i]
            x2[i] = x1[i] + sdir[i] * hs
        end
        fse[1] = call_integrand(impl_func, par, x1)
        fse[2] = call_integrand(impl_func, par, x2)
        fsum = fse[1] + fse[2]
        if fse[1] * fse[2] < 0
            xfs_pt.isc[1] = 1
            xfs_pt.isc[k + 2] = -1
            xfsl.isc[1] = 1
            xfsl.isc[k + 2] = -1
            inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsl,
                                                 baser, sdir, hs, nsub, xfsl.isc[k + 2])
            nsub += inters
            for i in 1:inters
                basei[nsub - i + 1] = 1
            end
            nbt[k + 1] = inters + 1
            if fse[1] < 0
                sign_sect[k + 1, 1] = -1
                sign_sect[k + 1, 2] = 1
            else
                sign_sect[k + 1, 1] = 1
                sign_sect[k + 1, 2] = -1
            end
        else
            nointer = true
            consi = vofi_check_side_consistency(ws, impl_func, par, x1, sdir, fse, hs)
            if consi != 0
                f2pos = consi
                sign_change = vofi_get_segment_min(ws, impl_func, par, x1, sdir,
                                                   fse, xfsl, hs, f2pos)
                if sign_change != 0
                    xfs_pt.isc[1] = 1
                    xfs_pt.isc[k + 2] = 1
                    xfsl.isc[1] = 1
                    xfsl.isc[k + 2] = 1
                    inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1,
                                                         xfsl, baser, sdir, hs,
                                                         nsub, xfsl.isc[k + 2])
                    nsub += inters
                    for i in 1:inters
                        basei[nsub - i + 1] = 1
                    end
                    nbt[k + 1] = inters + 1
                    xfs_pt.sval = 0.5 * (baser[nsub - 1] + baser[nsub - 2])
                    if fsum > 0
                        sign_sect[k + 1, 1] = 1
                        sign_sect[k + 1, 2] = -1
                        sign_sect[k + 1, 3] = 1
                    else
                        sign_sect[k + 1, 1] = -1
                        sign_sect[k + 1, 2] = 1
                        sign_sect[k + 1, 3] = -1
                    end
                    nointer = false
                end
            end
            if nointer
                atleast1 = true
                nbt[k + 1] = 1
                iside = k + 1
                isect = 2 - k
                sign_sect[k + 1, 1] = fsum > 0 ? 1 : (fsum < 0 ? -1 : 0)
                if (k == 0 && sign_sect[k + 1, 1] == 1) ||
                   (k == 1 && sign_sect[k + 1, 1] == -1)
                    down2up = -1
                end
            end
        end
    end

    ncheck = atleast1 ? max(nbt...) : 0
    baser[nsub + 1] = hs
    basei[nsub + 1] = 1
    vofi_reorder!(baser, basei, nsub + 1)
    nsub = vofi_rm_segs!(baser, basei, nsub)
    if ncheck == nsub
        vofi_sector_new!(sign_sect, nsect, ndire, nsub, iside, isect, down2up)
    else
        vofi_sector_old!(ws, impl_func, par, x0, h0, baser, pdir, sdir, nsect, ndire, nsub)
    end
    if nsub == 1 && ndire[1] == 0
        return 0
    end
    return nsub
end

function vofi_get_limits_inner_2D(ws::VofiWorkspace, impl_func, par, x0, h0, xfs_pt, base, pdir, sdir, nsect, ndire, nsub_int)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    basei = ws.gli_basei; fill!(basei, 0)
    baser[1] = 0
    basei[1] = 1
    nsub = 1
    hs = zero(T)
    for i in 1:NDIM
        hs += sdir[i] * h0[i]
    end
    x1 = ws.gli_x1
    x2 = ws.gli_x2
    fse = ws.gli_fse
    xfsl = ws.gli_xfsl
    copy!(xfsl, xfs_pt)

    for k in 0:1
        sign_change = true
        for i in 1:NDIM
            x1[i] = x0[i] + k * pdir[i] * h0[i]
            x2[i] = x1[i] + sdir[i] * hs
        end
        fse[1] = call_integrand(impl_func, par, x1)
        fse[2] = call_integrand(impl_func, par, x2)
        if fse[1] * fse[2] < 0
            inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsl,
                                                 baser, sdir, hs, nsub, -1)
            nsub += inters
            for i in 1:inters
                basei[nsub - i + 1] = 1
            end
        elseif xfsl.isc[k + 2] == 1
            for i in 1:NDIM
                xfsl.xval[i] = x1[i] + xfsl.sval * sdir[i]
            end
            fs = call_integrand(impl_func, par, xfsl.xval)
            xfsl.fval = fs
            fsum = fse[1] + fse[2]
            if fs * fsum >= 0
                f2pos = fsum > 0 ? 1 : (fsum < 0 ? -1 : 0)
                if f2pos != 0
                    sign_change = vofi_get_segment_min(ws, impl_func, par, x1, sdir, fse, xfsl, hs, f2pos) != 0
                end
            end
            if sign_change
                inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsl,
                                                     baser, sdir, hs, nsub, 1)
                nsub += inters
                xfs_pt.sval = 0.5 * (baser[nsub - 1] + baser[nsub - 2])
                for i in 1:inters
                    basei[nsub - i + 1] = 1
                end
            end
        end
    end

    baser[nsub + 1] = hs
    basei[nsub + 1] = 1
    vofi_reorder!(baser, basei, nsub + 1)
    nsub = vofi_rm_segs!(baser, basei, nsub)
    if nsub != nsub_int
        vofi_sector_old!(ws, impl_func, par, x0, h0, baser, pdir, sdir, nsect, ndire, nsub)
    end
    return nsub
end

function vofi_get_limits_edge_2D(ws::VofiWorkspace, impl_func, par, x0, h0, xfs_pt, base, pdir, sdir, nsub_int)
    T = promote_type(eltype(x0), eltype(h0))
    baser = base
    basei = ws.gle_basei; fill!(basei, 0)
    baser[1] = 0
    basei[1] = 1
    nsub = 1
    hs = zero(T)
    for i in 1:NDIM
        hs += sdir[i] * h0[i]
    end
    x1 = ws.gle_x1
    x2 = ws.gle_x2
    fse = ws.gle_fse
    xfsl = ws.gle_xfsl
    copy!(xfsl, xfs_pt)

    for k in 0:1
        sign_change = true
        for i in 1:NDIM
            x1[i] = x0[i] + k * pdir[i] * h0[i]
            x2[i] = x1[i] + hs * sdir[i]
        end
        fse[1] = call_integrand(impl_func, par, x1)
        fse[2] = call_integrand(impl_func, par, x2)
        if xfsl.isc[k + 2] == -1
            if fse[1] * fse[2] < 0
                inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsl,
                                                     baser, sdir, hs, nsub, -1)
            else
                vofi_check_edge_consistency(ws, impl_func, par, fse, x1, baser, sdir, hs, nsub)
            end
            basei[nsub + 1] = 1
            nsub += 1
        elseif xfsl.isc[k + 2] == 1
            for i in 1:NDIM
                xfsl.xval[i] = x1[i] + xfsl.sval * sdir[i]
            end
            fs = call_integrand(impl_func, par, xfsl.xval)
            xfsl.fval = fs
            fsum = fse[1] + fse[2]
            if fs * fsum >= 0
                f2pos = fsum > 0 ? 1 : (fsum < 0 ? -1 : 0)
                if f2pos != 0
                    sign_change = vofi_get_segment_min(ws, impl_func, par, x1, sdir,
                                                       fse, xfsl, hs, f2pos) != 0
                end
            end
            if !sign_change
                baser[nsub + 1] = xfsl.sval
                baser[nsub + 2] = xfsl.sval
                basei[nsub + 1] = 1
                basei[nsub + 2] = 1
                nsub += 2
            else
                inters = vofi_get_side_intersections(ws, impl_func, par, fse, x1, xfsl,
                                                     baser, sdir, hs, nsub, 1)
                nsub += inters
                for i in 1:inters
                    basei[nsub - i + 1] = 1
                end
            end
        end
    end

    baser[nsub + 1] = hs
    basei[nsub + 1] = 1
    vofi_reorder!(baser, basei, nsub + 1)
    #if nsub != nsub_int
    #    error("EXIT: vofi_get_limits_edge_2D unexpected subdivision count")
    #end
    return nsub
end
