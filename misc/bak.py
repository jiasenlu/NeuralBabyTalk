from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import math
import pdb

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, rnn_output, det_prob, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                             beam_pool_feats, beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, opt):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, \
                        beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs, rnn_output, beam_pnt_mask, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]

                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob })

            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()

            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()

                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()

                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()

            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                    beam_bn_seq[:t, vix] = beam_bn_seq_prev[:, v['q']]
                    beam_bn_seq_logprobs[:t, vix] = beam_bn_seq_logprobs_prev[:, v['q']]
                    beam_fg_seq[:t, vix] = beam_fg_seq_prev[:, v['q']]
                    beam_fg_seq_logprobs[:t, vix] = beam_fg_seq_logprobs_prev[:, v['q']]
                    beam_pnt_mask[:, vix] = beam_pnt_mask_prev[:, v['q']]

                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step

                new_rnn_output[vix] = rnn_output[v['q']] # dimension one is time step

                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

            state = new_state
            rnn_output = new_rnn_output

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs, \
                    beam_fg_seq, beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(), candidates

        # start beam search
        # opt = kwargs['opt']
        beam_size = opt.get('beam_size', 5)
        beam_att_mask = beam_pnt_mask.clone()
        rois_num = beam_ppls.size(1)

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()

        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
        done_beams = []
        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)
        logprobsf = {}
        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            det_prob = F.log_softmax(det_prob, dim=1)
            decoded = F.log_softmax(self.logit(rnn_output), dim=1)
            lambda_v = det_prob[:,0].contiguous()
            prob_det = det_prob[:,1:].contiguous()
            
            decoded = decoded + lambda_v.view(beam_size, 1).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)

            logprobsf[tag] = logprobs.data.cpu() # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            # logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000  
            beam_seq, beam_seq_logprobs,\
            beam_logprobs_sum,\
            beam_bn_seq, beam_bn_seq_logprobs, \
            beam_fg_seq, beam_fg_seq_logprobs, \
            rnn_output, state, beam_pnt_mask_new, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        beam_bn_seq, 
                                        beam_bn_seq_logprobs, 
                                        beam_fg_seq, 
                                        beam_fg_seq_logprobs,
                                        rnn_output,
                                        beam_pnt_mask_list[-1].t(),
                                        state)

            # encode as vectors
            it = beam_seq[t].cuda()
            roi_idx = it.clone() - self.vocab_size - 1 # starting from 0
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset 
            roi_idx_offset[roi_mask] = 0

            vis_idx = beam_ppls.data[:,:,4].contiguous().view(-1)[roi_idx_offset].long()
            vis_idx[roi_mask] = 0
            it_new = it.clone()
            it_new[it> self.vocab_size] = (vis_idx[roi_mask==0] + self.vocab_size)

            roi_labels = beam_pool_feats.data.new(beam_size*rois_num).zero_()
            if (roi_mask==0).sum() > 0: roi_labels[roi_idx_offset[roi_mask==0]] = 1
            roi_labels = roi_labels.view(beam_size, 1, rois_num)

            bn_logprob, fg_logprob = self.ccr_core(vis_idx, beam_pool_feats, \
                    rnn_output.view(beam_size,1,self.rnn_size), Variable(roi_labels), beam_size, 1)
            bn_logprob = bn_logprob.view(beam_size, -1)
            fg_logprob = fg_logprob.view(beam_size, -1)

            slp_bn, it_bn = torch.max(bn_logprob.data, 1)
            slp_fg, it_fg = torch.max(fg_logprob.data, 1)   

            it_bn[roi_mask] = 0
            it_fg[roi_mask] = 0

            beam_bn_seq[t] = it_bn
            beam_bn_seq_logprobs[t] = slp_bn

            beam_fg_seq[t] = it_fg
            beam_fg_seq_logprobs[t] = slp_fg

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix],
                        'bn_seq':beam_bn_seq[:, vix].clone(),
                        'bn_logps':beam_bn_seq_logprobs[:,vix].clone(),
                        'fg_seq':beam_fg_seq[:, vix].clone(),
                        'fg_logps':beam_fg_seq_logprobs[:,vix].clone(),
                    }
                    
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # updating the mask, and make sure that same object won't happen in the caption
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0
            beam_pnt_mask = beam_pnt_mask_new.data.clone()

            beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
            beam_pnt_mask.view(-1)[0] = 0
            beam_pnt_mask_list.append(Variable(beam_pnt_mask))

            xt = self.embed(Variable(it_new))
            rnn_output, det_prob, state = self.core(xt, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, beam_pool_feats, \
                                                            beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], state)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def constraint_beam_search(self, state, rnn_output, det_prob, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                             beam_pool_feats, beam_p_pool_feats, beam_ppls, beam_pnt_mask, vis_offset, roi_offset, tag_size, tags, opt):
        '''
        Implementation of the constraint beam search for image captioning.
        '''
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def constraint_beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, \
                        beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs, rnn_output, beam_pnt_mask, tags, vocab_size, tag_num, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            tag_list = range(vocab_size+1,vocab_size+tag_num+1)
            # create the hash table.
            if t == 0:
                for keys, logprobs in logprobsf.items():
                    logprobsf[keys] = logprobs.view(1, beam_size, -1)

            # candidates = []
            # rows = beam_size
            candidates = {tag:[] for tag in tags}
            # at beginning, the FSM state is just 0
            if t == 0: num_fsm = 1
            else: num_fsm = len(tags)

            for s in range(num_fsm): # for each FSM state.
                ys,ix = torch.sort(logprobsf[tags[s]],2,True)
                cols = min(beam_size, ys.size(2))

                if t == 0: rows = 1
                else: rows = torch.sum(beam_seq[0][s] != 0) # number of non-zero at first element.

                for q in range(rows):
                    tagSet = utils.containSet(tags, tags[q])

                    for tag in tagSet: # for each word
                        if tag == tags[q]: # if the tag is itself
                            # get the largest cols words that do not belong to the tag set.
                            tmpIdx = []
                            ii = 0
                            while len(tmpIdx) < cols:
                                if ix[s,q,ii] not in tag_list:
                                    tmpIdx.append(ii)
                                    ii += 1

                            for c in range(cols): # for each column (word, essentially)
                                # find the largest value that not belong to the tag set.
                                local_logprob = ys[s,q,tmpIdx[c]]
                                cc = ix[s,q,tmpIdx[c]]

                                candidate_logprob = beam_logprobs_sum[s, q] + local_logprob
                                candidates[tag].append({'c':cc, 'q':q, 'p':candidate_logprob, 'r':local_logprob })
                        else: # this is just for the transition of the FSM state.
                            # get the index
                            tag_diff = set(tag) - set(tags[q])
                            for tag_idx in tag_diff:
                                local_logprob = logprobsf[s, q, tag_idx+vocab_size+1]
                                cc = tag_idx + vocab_size + 1
                            
                            candidate_logprob = beam_logprobs_sum[s, q] + local_logprob
                            candidates[tag].append({'c':cc, 'q':q, 'p':candidate_logprob, 'r':local_logprob })
            for tag, candidate in candidates.items():
                candidates[tag] = sorted(candidate,  key=lambda x: -x['p'])
            
            pdb.set_trace()
            new_state = [_.clone() for _ in state]
            new_rnn_output = rnn_output.clone()

            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()

                beam_bn_seq_prev = beam_bn_seq[:t].clone()
                beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()

                beam_fg_seq_prev = beam_fg_seq[:t].clone()
                beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()
                
                beam_pnt_mask_prev = beam_pnt_mask.clone()
                beam_pnt_mask = beam_pnt_mask.clone()

            ######################################################################33
            # we are coding here
            ########################################################################
            
            for vix, tag in enumerate(tags):
                v = candidates[tag]
                pdb.set_trace()
                if len(v) != 0:
                    v = v[0]
                    #fork beam index q into index vix
                    if t >= 1:
                        beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                        beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                        beam_bn_seq[:t, vix] = beam_bn_seq_prev[:, v['q']]
                        beam_bn_seq_logprobs[:t, vix] = beam_bn_seq_logprobs_prev[:, v['q']]
                        beam_fg_seq[:t, vix] = beam_fg_seq_prev[:, v['q']]
                        beam_fg_seq_logprobs[:t, vix] = beam_fg_seq_logprobs_prev[:, v['q']]
                        beam_pnt_mask[:, vix] = beam_pnt_mask_prev[:, v['q']]

                    #rearrange recurrent states
                    for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                        new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step

                    new_rnn_output[vix] = rnn_output[v['q']] # dimension one is time step

                    #append new end terminal at the end of this beam
                    beam_seq[t, vix] = v['c'] # c'th word is the continuation
                    beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                    beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

            state = new_state
            rnn_output = new_rnn_output


            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs, \
                    beam_fg_seq, beam_fg_seq_logprobs, rnn_output, state, beam_pnt_mask.t(), candidates

        beam_size = opt.get('beam_size', 5)
        rois_num = beam_ppls.size(1)
        beam_att_mask = beam_pnt_mask.clone()
        fsm_num = len(tags)

        beam_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num, beam_size).zero_()
        beam_bn_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num, beam_size).zero_()
        beam_fg_seq = torch.LongTensor(self.seq_length, fsm_num, beam_size).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(self.seq_length, fsm_num, beam_size).zero_()

        beam_logprobs_sum = torch.zeros(fsm_num, beam_size) # running sum of logprobs for each beam
        done_beams = {tag:[] for tag in tags}

        beam_pnt_mask_list = []
        beam_pnt_mask_list.append(beam_pnt_mask)

        # insert the caption info to intial location.
        state = {():state}
        det_prob = {():det_prob}
        rnn_output = {():rnn_output}
        logprobsf = {}
        for t in range(self.seq_length):
            for tag in state.keys():
                """pem a beam merge. that is,
                for every previous beam we now many new possibilities to branch out
                we need to resort our beams to maintain the loop invariant of keeping
                the top beam_size most likely sequences."""
                det_prob[tag] = F.log_softmax(det_prob[tag], dim=1)
                decoded = F.log_softmax(self.logit(rnn_output[tag]), dim=1)
                lambda_v = det_prob[tag][:,0].contiguous()
                prob_det = det_prob[tag][:,1:].contiguous()
                
                decoded = decoded + lambda_v.view(beam_size, 1).expand_as(decoded)
                logprobs = torch.cat([decoded, prob_det], 1)

                logprobsf[tag] = logprobs.data.cpu() # lets go to CPU for more efficiency in indexing operations
                # suppress UNK tokens in the decoding
                # logprobsf[:,logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000
                
                beam_seq, beam_seq_logprobs,\
                beam_logprobs_sum,\
                beam_bn_seq, beam_bn_seq_logprobs, \
                beam_fg_seq, beam_fg_seq_logprobs, \
                rnn_output, state, beam_pnt_mask_new, \
                candidates_divm = constraint_beam_step(logprobsf,
                                            beam_size,
                                            t,
                                            beam_seq,
                                            beam_seq_logprobs,
                                            beam_logprobs_sum,
                                            beam_bn_seq, 
                                            beam_bn_seq_logprobs, 
                                            beam_fg_seq, 
                                            beam_fg_seq_logprobs,
                                            rnn_output,
                                            beam_pnt_mask_list[-1].t(),
                                            tags,
                                            self.vocab_size,
                                            tag_size,
                                            state)

                # encode as vectors
                it = beam_seq[t].cuda()
                roi_idx = it.clone() - self.vocab_size - 1 # starting from 0
                roi_mask = roi_idx < 0
                roi_idx_offset = roi_idx + vis_offset 
                roi_idx_offset[roi_mask] = 0

                vis_idx = beam_ppls.data[:,:,4].contiguous().view(-1)[roi_idx_offset].long()
                vis_idx[roi_mask] = 0
                it_new = it.clone()
                it_new[it> self.vocab_size] = (vis_idx[roi_mask==0] + self.vocab_size)

                roi_labels = beam_pool_feats.data.new(beam_size*rois_num).zero_()
                if (roi_mask==0).sum() > 0: roi_labels[roi_idx_offset[roi_mask==0]] = 1
                roi_labels = roi_labels.view(beam_size, 1, rois_num)

                bn_logprob, fg_logprob = self.ccr_core(vis_idx, beam_pool_feats, \
                        rnn_output.view(beam_size,1,self.rnn_size), Variable(roi_labels), beam_size, 1)

                bn_logprob = bn_logprob.view(beam_size, -1)
                fg_logprob = fg_logprob.view(beam_size, -1)

                slp_bn, it_bn = torch.max(bn_logprob.data, 1)
                slp_fg, it_fg = torch.max(fg_logprob.data, 1)   

                it_bn[roi_mask] = 0
                it_fg[roi_mask] = 0


                beam_bn_seq[t] = it_bn
                beam_bn_seq_logprobs[t] = slp_bn

                beam_fg_seq[t] = it_fg
                beam_fg_seq_logprobs[t] = slp_fg

                for vix in range(beam_size):
                    # if time's up... or if end token is reached then copy beams. 
                    # we don't want the last token to be the constraint word. This situation usually means we didn't
                    # find the best composition of the word.

                    if beam_seq[0, vix] != 0 and (beam_seq[t, vix] == 0 or t == self.seq_length - 1):
                        constraint_word = tags[vix]
                        skip_flag = False
                        for ii in constraint_word:
                            idx = ii + self.vocab_size + 1
                            idx_0 = 0
                            # find the first 0 or the maximum length
                            for jj in range(self.seq_length):
                                if beam_seq[jj,vix] == 0:
                                    idx_0 = jj
                                    break
                            if idx_0 == 0: idx_0 = self.seq_length

                            if idx == beam_seq[idx_0-1, vix] or idx == beam_seq[idx_0-2, vix]: #last one word or last two word
                                skip_flag = True

                        if skip_flag == False:
                            final_beam = {
                                'seq': beam_seq[:, vix].clone(),
                                'logps': beam_seq_logprobs[:, vix].clone(),
                                'p': beam_logprobs_sum[vix],
                                'bn_seq':beam_bn_seq[:, vix].clone(),
                                'bn_logps':beam_bn_seq_logprobs[:,vix].clone(),
                                'fg_seq':beam_fg_seq[:, vix].clone(),
                                'fg_logps':beam_fg_seq_logprobs[:,vix].clone(), 
                            }
                            done_beams[tags[vix]].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum[vix] = -1000

                # updating the mask, and make sure that same object won't happen in the caption
                pnt_idx_offset = roi_idx + roi_offset + 1
                pnt_idx_offset[roi_mask] = 0
                beam_pnt_mask = beam_pnt_mask_new.data.clone()

                beam_pnt_mask.view(-1)[pnt_idx_offset] = 1
                beam_pnt_mask.view(-1)[0] = 0
                beam_pnt_mask_list.append(Variable(beam_pnt_mask))

                xt = self.embed(Variable(it_new))
                rnn_output, det_prob, state = self.core(xt, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, beam_pool_feats, \
                                                                beam_p_pool_feats, beam_att_mask, beam_pnt_mask_list[-1], state)

        for tag, beams in done_beams.items():
            done_beams[tag] = sorted(beams,  key=lambda x: -x['p'])

        return done_beams