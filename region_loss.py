import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    #print("",.shape)
    print("target",target.shape)
    #target has shape [nBatch,5*50] eg [32,250]
    nB = target.size(0)
    print("nB",nB)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    #calc the total number of anchors in the grid. eg 5*13*13 = 845
    nAnchors = nA*nH*nW
    #calc number of cells in the grid eg 13*13
    nPixels  = nH*nW

    print("grid size nH,nW",(nH,nW))
    

    #for each sample in the batch. eg 32
    for b in range(nB):

        #get the current predicionts from the model. Shape [4,anchors*grid_w*grid_h] .eg [4,5*13*13] = [4,845]
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        print("cur_pred_boxes",cur_pred_boxes.shape)
        cur_ious = torch.zeros(nAnchors) #shape [anchors*grid_w*grid_h] .eg [5*13*13] = [845]

        #for all possible ground truth targets in this image. Its hard coded to be a max of 50 ground truth targets per image
        for t in range(50):
            if target[b][t*5+1] == 0:
                break

            #get the box coordinates for this target in this image. Scale to grid_w and grid_h
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH

            print("target x,y,w,x",(gx,gy,gw,gh))

            #repeat this targets coordinates across to create a tensor the same shape as the predicted box tensor eg [4,845]
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            print("cur_gt_boxes",cur_gt_boxes.shape)

            #for every prediction in the image, progresivly find the best IOU . eg shape [845] 
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            print("cur_ious",cur_ious.shape)
            print("conf_mask",conf_mask.shape)
        
        #reshape the list of ious to [anchors,height,width] eg [5,13,13]
        cur_ious = cur_ious.view(nA, nH, nW)

        #Use the ious to make binary mask for this image b in the batch. conf_mask shape = [32,5,13,13]
        #sil_thresh is hard coded at 0.6. Who knows why
        conf_mask[b][cur_ious>sil_thresh] = 0

    #some sort of warm up stuff
    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    #for every image in the batch
    for b in range(nB):

        #for every target for this image
        for t in range(50):

            #ensure there is at least one target for this image
            if target[b][t*5+1] == 0:
                break

            
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000

            #get the box coordinates for this target in this image. Scale to grid_w and grid_h
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gw = target[b][t*5+3] * nW
            gh = target[b][t*5+4] * nH

            #create ground truth box with just with and height to see which anchor has the best shape
            gt_box = [0, 0, gw, gh]

            #get the cell index for where this target box lands
            gi = int(gx)
            gj = int(gy)

            print("anchor_step",anchor_step)

            #Find the anchor size that best matches this ground truth size. eg 5 anchors
            for n in range(nA):
                #get the width and height from the anchor array 
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]

                #creat anchor with just width and height
                anchor_box = [0, 0, aw, ah]

                #check how well the groud truth size matches the anchor size
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)

                #false, our anchor step is 2
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)

                #keep track of the best matching anchor
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

                #false, our anchor step is 2
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            #create the ground truth box
            gt_box = [float(gx), float(gy), float(gw), float(gh)]

            #Use the best anchor index and the grid cell indexes to get the predicted box that will be matched with this target
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
            print("pred_boxes",pred_boxes.shape)
            print("pred_box",pred_box.shape)
            
            print("object_scale",object_scale)

            #activate the gradient for this prediction by setting the masks to one
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1

            #conf_mask has three values:
            #            0: predicted box overlaps ground truth but its anchor is not the best. The 0 ensures there is no gradient for this box
            #            1: this predicted box does not overlap the ground truth. The 1 ensure it gets a gradient that should supress this box
            # object_scale: eg 5. The anchor for this box has the best overlap for the ground truth. The 5 ensures that it gets a gradient boost 
            conf_mask[b][best_n][gj][gi] = object_scale


            #scale the target box to the grid size then subtract the cell indexs so the tx and ty are between 0-1 of their cell
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            
            #the model needs to predict the log of a scalar of the anchor size
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])

            #the target confidence is the IOU between the predicted box with the best anchor iou and the ground truth
            
            print("gt_box",gt_box)
            print("pred_box",pred_box)
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou

            #the class integer
            print("cls",target[b][t*5])
            tcls[b][best_n][gj][gi] = target[b][t*5]

            #if the iou is high enough count this as correct
            if iou > 0.5:
                nCorrect = nCorrect + 1
            

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors #array of floats eg [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
        self.num_anchors = num_anchors # eg 5
        self.anchor_step = len(anchors)//num_anchors # The number of values used to specifiy an anchor. eg (width,height) 10 // 5 = 2. 
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        #Split up the big model output into its various parts
        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))

        #view classes as [batch*anchors*grid_h*grid*w, num_Classes] eg [27040, 20]
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        #create a big tensor to hold all the bounding box cooordinates predicted for this batch [4, 27040]
        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)

        #create a kind of meshgrid for the cell indexes
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()

        #build a big tensor of anchors for every cell in every batch
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

        #populate the prediciton box tensor with the coordinates of each bounding box
        pred_boxes[0] = x.data.view(nB*nA*nH*nW) + grid_x
        pred_boxes[1] = y.data.view(nB*nA*nH*nW) + grid_y
        pred_boxes[2] = torch.exp(w.data.view(nB*nA*nH*nW)) * anchor_w
        pred_boxes[3] = torch.exp(h.data.view(nB*nA*nH*nW)) * anchor_h

        #put boxes on cpu
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        #buld the target tensors for this batch
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        
        #make this a binary mask
        cls_mask = (cls_mask == 1)


        nProposals = int((conf > 0.25).sum())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        
        #get all the class indexes for cells that have been assigned a ground truth by using the class mask
        tcls  = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())
        print("tcls",tcls)

        coord_mask = Variable(coord_mask.cuda())
        #TODO not sure what the square root does
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        #get the class logits for the cells that have been assigned a ground truth by using the class mask
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        #compute individual losses
        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0

        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0

        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls, loss))
        return loss
